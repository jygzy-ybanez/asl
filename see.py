import os
import numpy as np

DATA_PATH = os.path.join('MP_Data')
actions = np.array(sorted([f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]))
print("THIS IS YOUR ACTUAL ORDER:")
for i, action in enumerate(actions):
    print(f"Index {i} = Letter {action}")