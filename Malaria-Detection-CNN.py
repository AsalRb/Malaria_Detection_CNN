# Asal Rabiee
# Malaria Detection CNN

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Set the numpy pseudo-random generator at a fixed value
np.random.seed(1000)

# Set backend for Keras
os.environ['KERAS_BACKEND'] = 'tensorflow'

# ==============================================
# SET PATH

# Change this to your actual path
image_directory = r'C:\Users\Asal\Desktop\cell_images'

print(f" Using dataset from: {image_directory}")

# Verify the directory exists
if not os.path.exists(image_directory):
    print(f" ERROR: Directory not found: {image_directory}")
    print("Please check the path and make sure:")
    print("1. The folder 'cell_images' exists on your Desktop")
    print("2. It contains 'Parasitized' and 'Uninfected' subfolders")
    print("3. The path is spelled correctly")
    exit()

# Verify subdirectories exist
parasitized_path = os.path.join(image_directory, 'Parasitized')
uninfected_path = os.path.join(image_directory, 'Uninfected')

if not os.path.exists(parasitized_path):
    print(f" ERROR: Parasitized folder not found at: {parasitized_path}")
    exit()

if not os.path.exists(uninfected_path):
    print(f" ERROR: Uninfected folder not found at: {uninfected_path}")
    exit()

print(f"Parasitized folder found: {parasitized_path}")
print(f"Uninfected folder found: {uninfected_path}")

# Check how many images are in each folder
parasitized_files = [f for f in os.listdir(parasitized_path) if f.endswith('.png')]
uninfected_files = [f for f in os.listdir(uninfected_path) if f.endswith('.png')]

print(f"\n Found {len(parasitized_files)} parasitized images")
print(f" Found {len(uninfected_files)} uninfected images")

# ==============================================
# LOAD AND PREPROCESS IMAGES

SIZE = 64
dataset = []
label = []

print("\n Loading Parasitized images (label: 0)...")
for i, image_name in enumerate(parasitized_files):
    if i % 100 == 0 and i > 0:
        print(f"  Processed {i} parasitized images...")

    image_path = os.path.join(parasitized_path, image_name)
    image = cv2.imread(image_path)

    if image is not None:
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)
    else:
        print(f" Could not read image: {image_name}")

print("\n Loading Uninfected images (label: 1)...")
for i, image_name in enumerate(uninfected_files):
    if i % 100 == 0 and i > 0:
        print(f"  Processed {i} uninfected images...")

    image_path = os.path.join(uninfected_path, image_name)
    image = cv2.imread(image_path)

    if image is not None:
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)
    else:
        print(f" Could not read image: {image_name}")

print(f"\n Successfully loaded {len(dataset)} images")
print(f"   Parasitized (0): {label.count(0)}")
print(f"   Uninfected (1): {label.count(1)}")

if len(dataset) == 0:
    print(" ERROR: No images were loaded. Please check:")
    print("1. Images are in PNG format")
    print("2. Images are readable")
    print("3. File paths are correct")
    exit()

# ==============================================
# BUILD CNN MODEL

print("\n" + "=" * 50)
print("BUILDING CNN MODEL")
print("=" * 50)

INPUT_SHAPE = (SIZE, SIZE, 3)
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis=-1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)
conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis=-1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

flat = keras.layers.Flatten()(drop2)

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis=-1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)
hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis=-1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)

out = keras.layers.Dense(2, activation='softmax')(drop4)  

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# ==============================================
# PREPARE DATA FOR TRAINING

print("\n" + "=" * 50)
print("PREPARING DATA FOR TRAINING")
print("=" * 50)

# Convert to numpy arrays and normalize
X = np.array(dataset)
y = to_categorical(np.array(label))

# Normalize pixel values to [0, 1]
X = X / 255.0

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=0, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# ==============================================
# TRAIN THE MODEL

print("\n" + "=" * 50)
print("TRAINING THE MODEL")
print("=" * 50)

# Add callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )
]

history = model.fit(X_train,
                    y_train,
                    batch_size=32,  
                    verbose=1,
                    epochs=10,  
                    validation_split=0.1,
                    shuffle=True,  
                    callbacks=callbacks)

# ==============================================
# EVALUATE THE MODEL

print("\n" + "=" * 50)
print("EVALUATING THE MODEL")
print("=" * 50)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f" Test Accuracy: {test_accuracy:.2%}")
print(f" Test Loss: {test_loss:.4f}")



# ==============================================
# SAVE THE MODEL

print("\n" + "=" * 50)
print("SAVING THE MODEL")
print("=" * 50)

model_save_path = 'malaria_cnn_classifier.h5'
model.save(model_save_path)
print(f" Model saved to: {os.path.abspath(model_save_path)}")


