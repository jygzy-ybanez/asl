import numpy as np
import os
from tensorflow.keras.utils import to_categorical

DATA_PATH = 'MP_Data'
actions = np.array(sorted([f for f in os.listdir(DATA_PATH)]))
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for seq in range(30):
        window = [np.load(os.path.join(DATA_PATH, action, str(seq), f"{f}.npy")) for f in range(30)]
        sequences.append(window)
        labels.append(label_map[action])

np.save('X_data.npy', np.array(sequences))
np.save('y_labels.npy', to_categorical(labels).astype(int))
np.save('classes.npy', actions)
print("Data Packed!")