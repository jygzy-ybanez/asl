import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "data", "WLASL_parsed_data.json")

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

print(f"Type of JSON root: {type(data)}")

if isinstance(data, list):
    print(f"List length: {len(data)}")
    print("First item keys:", data[0].keys() if len(data) > 0 else "Empty List")
    if len(data) > 0:
        print("Sample item:", data[0])
elif isinstance(data, dict):
    print("Root keys:", data.keys())
    # If it's a dict, let's see the first key's content
    first_key = list(data.keys())[0]
    print(f"First key '{first_key}' content type: {type(data[first_key])}")