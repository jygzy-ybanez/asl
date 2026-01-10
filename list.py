import numpy as np

# Load the classes saved during training
classes = np.load('classes.npy')

print(f"Your model can recognize {len(classes)} unique words:")
print("-" * 30)
for i, word in enumerate(sorted(classes)):
    print(f"{i+1}. {word}")