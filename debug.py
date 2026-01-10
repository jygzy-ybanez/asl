import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load files
data = np.load(os.path.join(DATA_DIR, "landmarks_V3.npz"), allow_pickle=True)
with open(os.path.join(DATA_DIR, "WLASL_parsed_data.json"), 'r') as f:
    metadata = json.load(f)

# Print IDs from Landmarks file
print("IDs in .npz file (first 5):", list(data.keys())[:5])

# Print IDs from JSON file
json_ids = []
for entry in metadata:
    if 'instances' in entry:
        for inst in entry['instances']:
            json_ids.append(inst['video_id'])
            if len(json_ids) >= 5: break
    if len(json_ids) >= 5: break
print("IDs in JSON file (first 5):", json_ids)