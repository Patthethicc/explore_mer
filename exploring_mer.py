import cv2
import numpy as np
import sounddevice as sd
import torch
import queue
import threading
import time
import mediapipe as mp

from deepface import DeepFace
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ======================
# AUDIO EMOTION MODEL
# ======================
class SpeechEmotionModel:
    def __init__(self, model_name="superb/hubert-large-superb-er", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feat = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, wav, sr=16000):
        inputs = self.feat(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self.id2label[idx].lower(), float(probs[idx]), {self.id2label[i].lower(): float(p) for i, p in enumerate(probs)}

# ======================
# AUDIO STREAM RECORDER
# ======================
class AudioStream:
    def __init__(self, samplerate=16000, blocksize=16000, mic_index=None):
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
            print("[Audio Warning]", status)
        self.q.put(indata[:, 0].copy())

    def start(self): self.stream.start()
    def stop(self): self.stream.stop()

    def read_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

# ======================
# HAND GESTURE → EMOTION
# ======================
mp_hands = mp.solutions.hands
hands_processor = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def get_gesture_emotion(hand_landmarks):
    try:
        # Landmark indices
        THUMB_TIP = 4
        INDEX_TIP = 8
        INDEX_PIP = 6
        MIDDLE_TIP = 12
        MIDDLE_PIP = 10
        RING_TIP = 16
        RING_PIP = 14
        PINKY_TIP = 20
        PINKY_PIP = 18

        lm = hand_landmarks.landmark

        # Open Palm: All fingers extended
        open_palm = (lm[INDEX_TIP].y < lm[INDEX_PIP].y and
                     lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y and
                     lm[RING_TIP].y < lm[RING_PIP].y and
                     lm[PINKY_TIP].y < lm[PINKY_PIP].y)

        # Fist: All fingers curled
        fist = (lm[INDEX_TIP].y > lm[INDEX_PIP].y and
                lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y and
                lm[RING_TIP].y > lm[RING_PIP].y and
                lm[PINKY_TIP].y > lm[PINKY_PIP].y)

        # Thumbs Up
        thumb_up = (lm[THUMB_TIP].y < lm[INDEX_PIP].y and
                    lm[INDEX_TIP].y > lm[INDEX_PIP].y and
                    lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y)

        # Thumbs Down
        thumb_down = (lm[THUMB_TIP].y > lm[INDEX_PIP].y and
                      lm[INDEX_TIP].y > lm[INDEX_PIP].y and
                      lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y)

        # Rock & Roll
        rock_n_roll = (lm[INDEX_TIP].y < lm[INDEX_PIP].y and
                       lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y and
                       lm[RING_TIP].y > lm[RING_PIP].y and
                       lm[PINKY_TIP].y < lm[PINKY_PIP].y)

        # Map to Emotion
        if open_palm:      return "neutral", 1.0
        elif fist:         return "angry", 1.0
        elif thumb_up:     return "happy", 1.0
        elif thumb_down:   return "sad", 1.0
        elif rock_n_roll:  return "happy", 1.0  # Rock on = happy!
        else:              return "unknown", 0.0

    except Exception as e:
        print("[Gesture Error]", e)
        return "error", 0.0

# ======================
# FUSION LOGIC — 3 MODALITIES
# ======================
def fuse_triple_emotions(face_probs: dict, speech_probs: dict, gesture_emotion: str, gesture_conf: float):
    # Map face emotions to 4 core
    face_map = {
        "angry": "angry", "disgust": "angry", "fear": "sad",
        "happy": "happy", "sad": "sad", "surprise": "happy", "neutral": "neutral"
    }
    face_core = {"angry": 0.0, "happy": 0.0, "sad": 0.0, "neutral": 0.0}
    for emo, prob in face_probs.items():
        mapped = face_map.get(emo, "neutral")
        face_core[mapped] += prob
    total = sum(face_core.values()) or 1.0
    face_core = {k: v / total for k, v in face_core.items()}

    # Align speech
    speech_core = {
        "angry": speech_probs.get("angry", 0.0),
        "happy": speech_probs.get("happy", 0.0),
        "sad": speech_probs.get("sad", 0.0),
        "neutral": speech_probs.get("neutral", 0.0),
    }

    # Gesture as one-hot
    gesture_core = {"angry": 0.0, "happy": 0.0, "sad": 0.0, "neutral": 0.0}
    if gesture_emotion in gesture_core:
        gesture_core[gesture_emotion] = gesture_conf if gesture_conf > 0 else 1.0

    # Simple average fusion
    fused = {
        k: (face_core[k] + speech_core[k] + gesture_core[k]) / 3.0
        for k in face_core.keys()
    }

    dominant = max(fused.items(), key=lambda x: x[1])[0]
    return dominant.upper(), fused

# ======================
# MAIN APPLICATION
# ======================
def main():
    # --- Audio Setup ---
    print("Available audio devices:")
    print(sd.query_devices())
    mic_index_input = input("Enter microphone index (or press Enter for default): ").strip()
    mic_index = int(mic_index_input) if mic_index_input.isdigit() else None

    print("Loading speech emotion model...")
    speech_model = SpeechEmotionModel()
    audio_stream = AudioStream(mic_index=mic_index)
    audio_stream.start()

    # --- Video Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # --- Shared State ---
    face_label = "analyzing..."
    face_probs = {"neutral": 1.0}
    speech_label = "analyzing..."
    speech_conf = 0.0
    speech_probs = {"neutral": 1.0}
    gesture_label = "no hand"
    gesture_conf = 0.0

    # --- Audio Thread ---
    def audio_worker():
        nonlocal speech_label, speech_conf, speech_probs
        while True:
            wav = audio_stream.read_nowait()
            if wav is None:
                time.sleep(0.01)
                continue
            try:
                label, confidence, probs_dict = speech_model.predict(wav, sr=16000)
                speech_label, speech_conf, speech_probs = label, confidence, probs_dict
            except Exception as e:
                print("[Speech Error]", e)
                speech_label, speech_conf, speech_probs = "error", 0.0, {"neutral": 1.0}

    threading.Thread(target=audio_worker, daemon=True).start()

    # --- Main Loop ---
    print("Press 'q' to quit.")
    frame_counter = 0
    face_analysis_rate = 5  # every 5 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break

        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Facial Emotion (every N frames) ---
        if frame_counter % face_analysis_rate == 0:
            try:
                analysis = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(analysis, list):
                    analysis = analysis[0]

                raw_emotions = analysis.get("emotion", {})
                total = sum(raw_emotions.values()) or 1.0
                face_probs = {k: float(v)/total for k, v in raw_emotions.items()}
                face_label = max(face_probs, key=face_probs.get)

                # Draw face box
                if "region" in analysis:
                    x = analysis["region"]["x"]
                    y = analysis["region"]["y"]
                    w = analysis["region"]["w"]
                    h = analysis["region"]["h"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            except Exception as e:
                print("[Face Error]", e)
                face_label = "no face"
                face_probs = {"neutral": 1.0}

        # --- Hand Gesture ---
        gesture_label = "no hand"
        gesture_conf = 0.0
        results = hands_processor.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_label, gesture_conf = get_gesture_emotion(hand_landmarks)

        # --- Fuse All Three ---
        fused_label, _ = fuse_triple_emotions(face_probs, speech_probs, gesture_label, gesture_conf)

        # --- Overlay UI ---
        overlay_height = 200
        cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)

        cv2.putText(frame, f"Face: {face_label.capitalize()}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Speech: {speech_label.capitalize()} ({speech_conf:.2f})", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture_label.replace('_', ' ').title()}", 
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"FUSED: {fused_label}", 
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
        cv2.putText(frame, f"FUSED: {fused_label}", 
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

        # Display
        cv2.imshow('Trimodal Emotion Recognition (Face + Speech + Gesture)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    # Cleanup
    cap.release()
    audio_stream.stop()
    cv2.destroyAllWindows()
    print("Trimodal Emotion Recognition Terminated.")

if __name__ == "__main__":
    main()