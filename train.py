import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the data you just created
print("📂 Loading landmarks.csv...")
data = pd.read_csv("landmarks.csv", header=None)

# X = coordinates (63 columns), y = the letter (last column)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 2. Convert letters (A, B, C) into numbers (0, 1, 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# 3. Split: 80% to learn, 20% to test if it learned correctly
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Create the Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.2), # Prevents the model from "cheating" by memorizing
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax') # Outputs probabilities for each letter
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train the AI
print("⚙️ Training the AI... watch the 'accuracy' go up!")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 6. Save the results
model.save("gesture_model.h5")
np.save('classes.npy', label_encoder.classes_)
print("✅ Success! 'gesture_model.h5' is ready. This is your AI Brain.")