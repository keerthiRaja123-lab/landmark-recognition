import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

IMG_SIZE = 128

model = load_model("models/landmark_model.h5")
class_names = os.listdir("dataset/train")

img_path = input("Enter image path: ")

img = cv2.imread(img_path)

if img is None:
    print("Invalid image path!")
    exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted Landmark:", predicted_class)