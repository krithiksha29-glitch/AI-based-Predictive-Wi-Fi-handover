import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load your prepared data
X_train = np.load('X_train_scaled.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Build the LSTM model
model = keras.Sequential([
    layers.Input(shape=(5, 4)),          # 5 time-steps, 4 towers
    layers.LSTM(32, unroll=True),                      # the LSTM layer — 32 "memory units"
    layers.Dense(16, activation='relu'),  # a regular processing layer
    layers.Dense(4, activation='softmax') # output: probability for each of 4 towers
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# Save the trained model
model.save('handover_predictor.keras')
print("Model saved as handover_predictor.keras")