import os
import numpy as np
import json

# --- ADD THESE TWO LINES ---
# This defines SCRIPT_DIR as the folder where preprocess.py is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Now line 8 will work because SCRIPT_DIR exists
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
# ---------------------------

print(f"Project directory identified as: {SCRIPT_DIR}")
print(f"Looking for data in: {DATA_DIR}")

MAX_FRAMES = 30

def preprocess_sequences():
    # Load raw data
    data_v3 = np.load(os.path.join(DATA_DIR, "landmarks_V3.npz"), allow_pickle=True)
    with open(os.path.join(DATA_DIR, "WLASL_parsed_data.json"), "r") as f:
        metadata = json.load(f)

    X = [] # To store sequences
    y = [] # To store labels (glosses)

    for video_id in data_v3.files:
        sequence = data_v3[video_id] # Shape: (frames, 553, 3)
        
        # 1. SLICING: Keep only Hands (0-41) and upper Pose (42-48)
        # This reduces 553 points down to 48, making training much faster
        sequence = sequence[:, :48, :] 
        
        # 2. FLATTEN: Convert (frames, 48, 3) to (frames, 144)
        sequence = sequence.reshape(sequence.shape[0], -1)

        # 3. PADDING/TRUNCATING: Ensure exactly 30 frames
        if len(sequence) > MAX_FRAMES:
            sequence = sequence[:MAX_FRAMES] # Cut if too long
        else:
            # Add zeros at the end if too short
            padding = np.zeros((MAX_FRAMES - len(sequence), sequence.shape[1]))
            sequence = np.append(sequence, padding, axis=0)

        X.append(sequence)
        y.append(metadata[video_id]['gloss'])

    return np.array(X), np.array(y)

X, y = preprocess_sequences()
print(f"Final Data Shape: {X.shape}") # Expect (Total_Videos, 30, 144)