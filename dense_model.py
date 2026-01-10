import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten

X = np.load('X_data.npy')
y = np.load('y_labels.npy')
actions = np.load('classes.npy')

model = Sequential([
    Input(shape=(30, 144)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100)
model.save('asl_model.h5')