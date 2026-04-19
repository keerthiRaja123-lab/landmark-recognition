import os
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/train"
IMG_SIZE = 128

data = []
labels = []
class_names = os.listdir(DATASET_PATH)

print("Loading images...")

for label, folder in enumerate(class_names):
    folder_path = os.path.join(DATASET_PATH, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(label)

data = np.array(data) / 255.0
labels = to_categorical(labels)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print("Building model...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training started...")

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print("Final Accuracy:", acc)

model.save("models/landmark_model.h5")

print("Model saved successfully!")