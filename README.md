1. Data Setup
This repository contains the code logic only. To run the project, you must first download the dataset:

Download Link: [https://drive.google.com/drive/folders/1Gkj5oVIFq3mZYV9pAPzZ936AF5vv7mkn?usp=sharing]

Setup: Extract the data and ensure your folder follows this structure:
ARCHIVE/
├── data/
├── MP_Data/
├── .gitattributes
├── .gitignore
├── app.py
├── asl_model.h5
├── check_files.py
├── classes.npy
├── clean_collect.py
├── collect_alphabet.py
├── create_custom_data.py
├── data.zip
├── debug.py
├── dense_model.py
├── filtered_labels.txt
├── final_train.py
├── inspect_json.py
├── labels.npz
├── landmarks_V1.npz
├── landmarks_V2.npz
├── list.py
├── packer.py
├── preprocess_final.py
├── preprocess.py
├── README.md
├── see.py
├── test_fix.py
├── train_asl.py
├── X_data.npy
└── y_labels.npy

Note: If you notice any files mentioned in the scripts that are missing from this file structure, please let me know!

2. Installation & Dependencies
Run the application using the following command:
python app.py

Note on Errors: 
- If the program fails to start, it is likely due to missing libraries (e.g., opencv, tensorflow, or mediapipe). Since this project uses several dependencies, please search for any ModuleNotFoundError you receive and install the required package.

3. Usage
Once the environment is set up and all errors are resolved:
- A camera window will automatically pop up. (To exit camera window, click "Q")
- Position yourself in front of the camera to perform sign language gestures.
- The model will detect and translate your signs in real-time.

4. Supported Signs
The model is trained specifically on the gestures found at this link: [https://www.ai-media.tv/wp-content/uploads/ASL_Alphabet.jpg]. Please refer to this guide to see which signs the model recognizes.
