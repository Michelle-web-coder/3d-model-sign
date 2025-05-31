import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model and label encoder
with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_map.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# OpenCV webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])  # (x, y) only

            # Predict if 42 values (21 x 2)
            if len(landmarks) == 42:
                prediction = model.predict([landmarks])[0]
                predicted_label = label_encoder.inverse_transform([prediction])[0]

                # Display prediction
                cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
