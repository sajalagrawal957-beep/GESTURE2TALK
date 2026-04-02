# utils/predictor.py

import numpy as np
import tensorflow as tf

LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

class GesturePredictor:
    def __init__(self, model_path):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded!")

    def predict(self, landmarks):
        input_data = np.array(landmarks).reshape(1, -1)
        predictions = self.model.predict(input_data, verbose=0)
        best_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][best_index])
        letter = LABELS[best_index]
        return letter, confidence