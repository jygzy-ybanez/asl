import numpy as np
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.preprocessing import LabelEncoder

# 1. SETUP PATHS
# Dynamically find the path based on where this script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# 2. LOAD DATA
data = np.load(os.path.join(DATA_DIR, "landmarks_V3.npz"), allow_pickle=True)
with open(os.path.join(DATA_DIR, "WLASL_parsed_data.json"), 'r') as f:
    metadata = json.load(f)

# 3. PREPROCESS & FILTER
X = []
y_raw = []

# Using a subset of the first 500 samples for a quick test run
for key in list(data.keys())[:500]: 
    seq = data[key] # Sequence shape: (frames, 553, 3)
    
    # Filter: Keep first 48 landmarks (Hands + Pose)
    # This reduces 553 points to 48, speeding up training significantly
    filtered_seq = seq[:, :48, :] 
    
    # Flatten: (frames, 48, 3) -> (frames, 144)
    flattened = filtered_seq.reshape(filtered_seq.shape[0], -1)
    
    # Standardize to 30 frames (padding or truncating)
    if len(flattened) < 30:
        pad = np.zeros((30 - len(flattened), 144))
        flattened = np.vstack((flattened, pad))
    else:
        flattened = flattened[:30]
        
    X.append(flattened)
    y_raw.append(metadata[key]['gloss'])

# Convert to Numpy Arrays
X = np.array(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw) # Turns words like "book" into numbers

# 4. DEFINE THE AI MODEL
model = Sequential([
    Input(shape=(30, 144)), 
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])