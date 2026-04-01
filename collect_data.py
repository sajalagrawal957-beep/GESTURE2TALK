import cv2
import mediapipe as mp
import csv
import os

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

print("⌨️ Press the Letter key (A, B, C...) to save 100 samples of that sign.")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Key Press check
            key = cv2.waitKey(1) & 0xFF
            if key >= ord('a') and key <= ord('z'):
                label = chr(key).upper()
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                coords.append(label)
                
                # Append to your existing CSV
                with open("landmarks.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    for _ in range(50): # Save 50 copies of this exact frame to boost data
                        writer.writerow(coords)
                print(f"✅ Saved 50 samples for letter: {label}")

    cv2.imshow("Data Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()