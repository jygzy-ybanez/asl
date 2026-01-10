import numpy as np
import os
import json
import re # Added for flexible text searching

# Configuration
MAX_FRAMES = 30
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
ALPHABET = list("abcdefghijklmnopqrstuvwxyz") 
LIMIT_PER_LETTER = 100 

def run_preprocess():
    print("Loading raw data... Searching for ALL alphabet letters.")
    
    try:
        data = np.load(os.path.join(DATA_DIR, "landmarks_V3.npz"), allow_pickle=True)
        with open(os.path.join(DATA_DIR, "WLASL_parsed_data.json"), 'r') as f:
            metadata_list = json.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find file: {e}")
        return

    # 1. Build a more flexible ID-to-Gloss mapping
    id_to_gloss = {}
    for entry in metadata_list:
        path = entry.get('video_path', '')
        raw_gloss = entry.get('gloss', '').lower().strip()
        
        if path and raw_gloss:
            v_id = path.replace('\\', '/').split('/')[-1].replace('.mp4', '')
            
            # This regex looks for single letters even in strings like "a (letter)"
            # It ensures we don't accidentally grab "apple" when looking for "a"
            match = re.match(r'^([a-z])(\s+\(letter\))?$', raw_gloss)
            if match:
                id_to_gloss[v_id] = match.group(1)
            elif raw_gloss in ALPHABET:
                id_to_gloss[v_id] = raw_gloss

    X, y = [], []
    word_counts = {} 
    all_keys = list(data.keys())
    
    # 2. Process landmarks
    for v_id in all_keys:
        if v_id in id_to_gloss:
            gloss = id_to_gloss[v_id]
            
            if word_counts.get(gloss, 0) >= LIMIT_PER_LETTER:
                continue
            
            raw_seq = data[v_id]
            window = raw_seq[:, :48, :] 
            window = window.reshape(window.shape[0], -1)

            if len(window) > MAX_FRAMES:
                window = window[:MAX_FRAMES]
            else:
                pad = np.zeros((MAX_FRAMES - len(window), 144))
                window = np.vstack((window, pad))

            X.append(window)
            y.append(gloss)
            word_counts[gloss] = word_counts.get(gloss, 0) + 1

    if len(X) > 0:
        np.save(os.path.join(SCRIPT_DIR, 'X_data.npy'), np.array(X))
        np.save(os.path.join(SCRIPT_DIR, 'y_labels.npy'), np.array(y))
        print(f"\n[SUCCESS] Total Samples: {len(X)}")
        print(f"Letters Found: {len(word_counts)} / 26")
        print(f"Details: {dict(sorted(word_counts.items()))}")
    else:
        print("[FAILED] No alphabet samples found. The dataset might not contain alphabet landmarks.")

if __name__ == "__main__":
    run_preprocess()