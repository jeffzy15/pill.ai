import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

DATA_DIR = 'data'
IMG_SIZE = 50
MODEL_PATH = 'model.keras'
CATEGORIES = os.listdir(DATA_DIR)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model_path, image_path):
    model = load_model(model_path)
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    return CATEGORIES[class_index], confidence
