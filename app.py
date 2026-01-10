import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras.models import load_model

# 1. Load model
model = load_model('asl_model.h5')

# 2. AUTOMATICALLY load labels from your training data
# This prevents the "IndexError: index out of bounds"
if os.path.exists('classes.npy'):
    actions = np.load('classes.npy')
    print(f"Successfully loaded {len(actions)} classes: {actions}")
else:
    # Fallback to manual list if classes.npy is missing
    actions = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    print("Warning: classes.npy not found. Using fallback list.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        rh_raw = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
        # WRIST CENTERING
        rh = (rh_raw - rh_raw[0]).flatten() 
        return np.concatenate([rh, np.zeros(63), np.zeros(18)]) # Match 144 features
    return None

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        status_text = "Status: No Hand"

        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            keypoints = extract_keypoints(results)
            
            if keypoints is not None:
                # Mirror the frame to fill the 30-frame sequence the model expects
                input_data = np.tile(keypoints, (30, 1)) 
                input_data = np.expand_dims(input_data, axis=0)
                
                res = model.predict(input_data, verbose=0)[0]
                best_idx = np.argmax(res)
                
                # SAFETY CHECK: Ensure index exists in our actions list
                if best_idx < len(actions):
                    status_text = f"Guess: {actions[best_idx].upper()} | Conf: {res[best_idx]:.2f}"
                else:
                    status_text = f"Error: Predicted index {best_idx} not in list"

        # Blue Diagnostic Bar
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('ASL Alphabet System', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()