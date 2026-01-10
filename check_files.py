import numpy as np
import os

# Get the folder where check_files.py is actually saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path to landmarks_V3.npz inside the 'data' folder
DATA_PATH = os.path.join(BASE_DIR, "data", "landmarks_V3.npz")

print(f"Searching for file at: {DATA_PATH}")

try:
    # Load the file
    data = np.load(DATA_PATH, allow_pickle=True)
    
    # See the first 5 video IDs inside
    keys = list(data.keys())
    print(f"[OK] Success! Found {len(keys)} videos.")
    print("Example Video IDs:", keys[:5])
    
    # Check the shape of the first video
    first_key = keys[0]
    print(f"Shape of video {first_key}: {data[first_key].shape}")

except FileNotFoundError:
    print(f"[ERROR] Could not find the file at {DATA_PATH}")
    print("Check if your 'data' folder is actually inside the 'archive' folder.")
except Exception as e:
    print(f"[ERROR] An unexpected error occurred: {e}")