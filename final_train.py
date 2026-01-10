import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 1. Load Data
X = np.load('X_data.npy')
y = np.load('y_labels.npy')
actions = np.load('classes.npy')

print(f"Loading data for {len(actions)} letters...")

# 2. Split (Using 10% for test to better evaluate 26 letters)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# 3. Build the "Alphabet Pro" Model
# We use more neurons (512 and 256) to handle the 26-class complexity
model = Sequential([
    Input(shape=(30, 144)),
    Flatten(), 
    Dense(512, activation='relu'),
    Dropout(0.4), # Higher dropout to prevent the model from just memorizing your room
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

# 4. Compile with a lower learning rate
# 0.0001 is slower but prevents the model from "jumping" over the best solution
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 5. Train
# Increased epochs to 250 because 26 letters take longer to learn
print("Starting Training...")
model.fit(
    X_train, 
    y_train, 
    epochs=250, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    verbose=1
)

# 6. Save
model.save('asl_model.h5')
print("\n[SUCCESS] Full Alphabet Model Saved as asl_model.h5")