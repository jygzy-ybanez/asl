import cv2
import numpy as np
import os
import mediapipe as mp

ACTIONS = np.array(['a', 'b']) # Let's just fix A and B first
DATA_PATH = os.path.join('MP_Data')

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

def extract_centered_keypoints(results):
    if results.multi_hand_landmarks:
        res = results.multi_hand_landmarks[0]
        full_res = np.array([[l.x, l.y, l.z] for l in res.landmark])
        # CENTER ON WRIST: This is the magic fix
        centered = (full_res - full_res[0]).flatten()
        return np.concatenate([centered, np.zeros(63), np.zeros(18)]) # Match shape 144
    return np.zeros(144)

with mp_hands.Hands(min_detection_confidence=0.7) as hands:
    for action in ACTIONS:
        for seq in range(30):
            for frame_num in range(30):
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                
                # Visuals
                cv2.putText(frame, f'RECORDING {action.upper()} - Video {seq}', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow('Collector', frame)

                # SAVE
                keypoints = extract_centered_keypoints(results)
                path = os.path.join(DATA_PATH, action, str(seq))
                os.makedirs(path, exist_ok=True)
                np.save(os.path.join(path, str(frame_num)), keypoints)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()