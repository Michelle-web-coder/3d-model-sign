import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Settings
SAVE_DIR = "collected_data"
LABEL = "Z"  # Change this for each sign
SAMPLES_PER_LABEL = 200  # Number of samples to collect

# Create folder
os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, f"{LABEL}.csv")

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Capture from webcam
cap = cv2.VideoCapture(0)
sample_count = 0

with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    print(f"Collecting data for label '{LABEL}'. Press 'q' to stop.")

    while cap.isOpened() and sample_count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get 21 (x, y) coords normalized
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                if len(landmarks) == 42:  # 21 points x 2 (x and y)
                    writer.writerow([LABEL] + landmarks)
                    sample_count += 1

        cv2.putText(frame, f"Samples: {sample_count}/{SAMPLES_PER_LABEL}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("Data collection complete.")
cap.release()
cv2.destroyAllWindows()
