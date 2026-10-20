import numpy as np

# Load the train/test split from before
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')

# Step 1: Find the min and max — ONLY from training data
train_min = X_train.min()
train_max = X_train.max()

print("Training data min RSSI:", train_min)
print("Training data max RSSI:", train_max)

# Step 2: Apply min-max scaling to both train and test, using training min/max
X_train_scaled = (X_train - train_min) / (train_max - train_min)
X_test_scaled = (X_test - train_min) / (train_max - train_min)

print()
print("Before scaling — example value:", X_train[0][0][0])
print("After scaling  — example value:", X_train_scaled[0][0][0])

print()
print("Scaled training data range:", X_train_scaled.min(), "to", X_train_scaled.max())
print("Scaled testing data range:", X_test_scaled.min(), "to", X_test_scaled.max())

# Step 3: Save the scaled versions for model training
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)

# Also save the min/max — we'll need these later to "unscale" predictions back to real RSSI
np.save('scaling_params.npy', np.array([train_min, train_max]))

print()
print("Saved X_train_scaled.npy, X_test_scaled.npy, and scaling_params.npy")