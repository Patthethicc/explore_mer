import cv2
import numpy as np
import sounddevice as sd
import torch
import queue
import threading
import time

from deepface import DeepFace
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Audio Model
class SpeechEmotionModel:
    def __init__(self, model_name="superb/hubert-large-superb-er", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feat = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, wav, sr=16000):
        # expects mono float32 numpy (length ~ block of audio)
        inputs = self.feat(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self.id2label[idx].lower(), float(probs[idx]), probs, {self.id2label[i].lower(): float(p) for i, p in enumerate(probs)}

# Audio Recorder
class AudioStream:
    def __init__(self, samplerate=16000, blocksize=16000, mic_index=None):  # ~1 sec blocks
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.q = queue.Queue()

        if mic_index is not None:
            sd.default.device = (mic_index, None)

        self.stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            blocksize=blocksize,
            dtype="float32",
            callback=self.callback
        )

    def callback(self, indata, frames, time_, status):
        if status:
            print("[audio]", status)
        self.q.put(indata[:, 0].copy())

    def start(self): self.stream.start()
    def stop(self): self.stream.stop()

    def read_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

# Simple fusion
def fuse(face_probs: dict, speech_probs: dict):
    # map DeepFace's 7 emotions into {angry, happy, sad, neutral}
    map_face = {
        "angry": "angry",
        "disgust": "angry",
        "fear": "sad",
        "happy": "happy",
        "sad": "sad",
        "surprise": "happy",
        "neutral": "neutral",
    }
    base = {"angry": 0.0, "happy": 0.0, "sad": 0.0, "neutral": 0.0}
    for k, v in face_probs.items():
        base[map_face.get(k, "neutral")] += float(v)
    s = sum(base.values()) or 1.0
    face_shared = {k: v/s for k, v in base.items()}

    speech_shared = {
        "angry": speech_probs.get("angry", 0.0),
        "happy": speech_probs.get("happy", 0.0),
        "sad": speech_probs.get("sad", 0.0),
        "neutral": speech_probs.get("neutral", 0.0),
    }

    fused = {k: (face_shared[k] + speech_shared[k]) / 2.0 for k in base.keys()}
    top = max(fused.items(), key=lambda x: x[1])[0]
    return top, fused

# Main 
def main():
    # select input device
    print("Available audio devices:")
    print(sd.query_devices())
    mic_index = int(input("Enter the index of your microphone (or -1 for default): "))
    if mic_index == -1:     
        mic_index = None

    print("Loading models (first time may download weights)...")
    speech_model = SpeechEmotionModel()
    audio = AudioStream(mic_index=mic_index)
    audio.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    face_label = "neutral"
    speech_label = "neutral"
    speech_conf = 0.0
    speech_probs = {"neutral": 1.0}
    face_probs = {"neutral": 1.0}

    # run speech inference in background
    def audio_worker():
        nonlocal speech_label, speech_conf, speech_probs
        while True:
            wav = audio.read_nowait()
            if wav is None:
                time.sleep(0.01)
                continue
            print("[mic] got", len(wav), "samples")  # ✅ debug print
            try:
                lbl, prob, _, probs_dict = speech_model.predict(wav, sr=16000)
                speech_label, speech_conf, speech_probs = lbl, prob, probs_dict
            except Exception as e:
                print("[audio inference error]", e)

    threading.Thread(target=audio_worker, daemon=True).start()

    print("Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # facial expression detection
        try:
            analysis = DeepFace.analyze(
                frame,
                actions=["emotion"],
                detector_backend="opencv",   # faster, works well live
                enforce_detection=False,
                prog_bar=False
            )
            if isinstance(analysis, list):
                analysis = analysis[0]
            emos = analysis.get("emotion", {})
            tot = sum(float(v) for v in emos.values()) or 1.0
            face_probs = {k: float(v)/tot for k, v in emos.items()}
            face_label = max(face_probs.items(), key=lambda x: x[1])[0]

            # draw bounding box if available
            if "region" in analysis:
                x, y, w, h = analysis["region"].values()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        except Exception:
            face_label = "unknown"
            face_probs = {"neutral": 1.0}

        fused_label, fused_probs = fuse(face_probs, speech_probs)

        # draw overlay
        cv2.putText(frame, f"Face: {face_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Speech: {speech_label} ({speech_conf:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"FUSED: {fused_label.upper()}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
        cv2.putText(frame, f"FUSED: {fused_label.upper()}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

        cv2.imshow("Live Emotion Recognition (Face + Speech)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    audio.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
