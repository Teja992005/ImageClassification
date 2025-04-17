import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import os
import json

# Create 'model' directory if not exists
os.makedirs("model", exist_ok=True)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values (0 to 1 range)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define class labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Data Augmentation for better performance
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Train model
print("🚀 Training the model... This may take a few minutes.")
model.fit(datagen.flow(x_train, y_train, batch_size=64), 
          epochs=15, 
          validation_data=(x_test, y_test))

# Save model
model.save("model/model.h5")
print("✅ Model saved successfully at 'model/model.h5'!")

# Save class names
with open("model/labels.json", "w") as f:
    json.dump(CLASS_NAMES, f)
print("✅ Labels saved successfully at 'model/labels.json'!")

# Verify that the model is saved
if os.path.exists("model/model.h5"):
    print("🎉 Model verification successful!")
else:
    print("❌ Model file NOT found. Check the directory.")
