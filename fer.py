import cv2
from deepface import DeepFace

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# Frame management
frame_counter = 0
emotion_analysis_rate = 10  # analyze every N frames
current_emotion = "analyzing..."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    frame_counter += 1

    # DeepFace facial emotion
    if frame_counter % emotion_analysis_rate == 0:
        try:
            analysis = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                detector_backend="mediapipe",
                enforce_detection=False,
                silent=True
            )
            if isinstance(analysis, list) and len(analysis) > 0:
                current_emotion = analysis[0]['dominant_emotion']
        except Exception:
            current_emotion = "no face"

    # display text
    cv2.rectangle(frame, (0, 0), (400, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Facial Emotion: {current_emotion.capitalize()}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow('Real-time Emotion Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()