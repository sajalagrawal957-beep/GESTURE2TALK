import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the Brain and the Dictionary
model = load_model("gesture_model.h5")
classes = np.load('classes.npy', allow_pickle=True)

# 2. Setup Camera and Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("🚀 Webcam starting... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip the frame (like a mirror) and convert to RGB
    frame = cv2.flip(frame, 1)
    f_height, f_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the landmarks on the screen so you can see them
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract coordinates for the model
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            # Predict!
            prediction = model.predict(np.array([coords]), verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            label = classes[class_id]
            print(f"Detected: {label} | Confidence: {int(confidence*100)}%")

            # Display the result on the screen
            text = f"{label} ({int(confidence*100)}%)"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ASL Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()