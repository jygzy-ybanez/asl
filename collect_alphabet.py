import cv2
import numpy as np
import os
import mediapipe as mp

ACTIONS = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
DATA_PATH = 'MP_Data'

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        res = results.multi_hand_landmarks[0]
        full_res = np.array([[l.x, l.y, l.z] for l in res.landmark])
        # WRIST CENTERING: All points relative to point 0
        centered = (full_res - full_res[0]).flatten()
        # Padding to match 144 features (63 hand + 63 other hand + 18 pose)
        return np.concatenate([centered, np.zeros(63), np.zeros(18)])
    return np.zeros(144)

with mp_hands.Hands(min_detection_confidence=0.7) as hands:
    for action in ACTIONS:
        for seq in range(30):
            for frame_num in range(30):
                ret, frame = cap.read()
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                msg = f'STARTING {action.upper()}' if frame_num == 0 else f'Recording {action}'
                cv2.putText(frame, msg, (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow('OpenCV', frame)

                keypoints = extract_keypoints(results)
                path = os.path.join(DATA_PATH, action, str(seq))
                os.makedirs(path, exist_ok=True)
                np.save(os.path.join(path, str(frame_num)), keypoints)
                cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()