import numpy as np
import os
from tensorflow.keras.utils import to_categorical

# 1. Setup Paths
DATA_PATH = os.path.join('MP_Data') 
# This automatically picks up 'a', 'b', 'c' folders in order
actions = np.array(sorted([f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]))
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

# 2. Loop through folders and pack data
for action in actions:
    for sequence in range(30): # Number of videos per letter
        window = []
        for frame_num in range(30): # 30 frames per video
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# 3. Save the final files
X = np.array(sequences)
y = to_categorical(labels).astype(int)

np.save('X_data.npy', X)
np.save('y_labels.npy', y)
np.save('classes.npy', actions)

print(f"Successfully created X_data.npy and y_labels.npy for: {actions}")