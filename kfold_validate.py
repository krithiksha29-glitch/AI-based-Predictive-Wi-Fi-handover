import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---- Load the full (unsplit) dataset ----
X = np.load('X_data.npy')
y = np.load('y_data.npy')

df = pd.read_csv('pivoted_data.csv')
WINDOW_SIZE = 5

window_set_walk = []
for (set_num, walk_num), group in df.groupby(['set_number', 'walk_number']):
    group = group.reset_index(drop=True)
    n_windows_this_walk = len(group) - WINDOW_SIZE
    if n_windows_this_walk > 0:
        window_set_walk += [(set_num, walk_num)] * n_windows_this_walk
window_set_walk = np.array(window_set_walk)

# All unique (set, walk) combos — we fold by WALK, never splitting a walk across train/test
unique_walks = sorted(set(map(tuple, window_set_walk)))
print(f"Total walks: {len(unique_walks)}")
print(f"Total windows: {X.shape[0]}")

N_FOLDS = 5
np.random.seed(42)
shuffled_walks = unique_walks.copy()
np.random.shuffle(shuffled_walks)
folds = np.array_split(shuffled_walks, N_FOLDS)

num_features = X.shape[2]

fold_accuracies = []
for fold_idx, test_walks in enumerate(folds):
    test_walks = set(map(tuple, test_walks))
    test_mask = np.array([tuple(sw) in test_walks for sw in window_set_walk])
    train_mask = ~test_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Scale RSSI (cols 0-3) and accel (remaining cols, if any) separately, using TRAIN fold only
    rssi_min, rssi_max = X_train[:, :, 0:4].min(), X_train[:, :, 0:4].max()
    has_accel = X_train.shape[2] > 4
    if has_accel:
        accel_min, accel_max = X_train[:, :, 4:].min(), X_train[:, :, 4:].max()

    def scale(Xd):
        rssi = (Xd[:, :, 0:4] - rssi_min) / (rssi_max - rssi_min)
        if has_accel:
            accel = (Xd[:, :, 4:] - accel_min) / (accel_max - accel_min)
            return np.concatenate([rssi, accel], axis=2)
        return rssi

    X_train_s = scale(X_train)
    X_test_s = scale(X_test)

    model = keras.Sequential([
        layers.Input(shape=(5, num_features)),
        layers.LSTM(32, unroll=True),
        layers.Dense(16, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_s, y_train, epochs=30, batch_size=8, verbose=0)
    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)

    print(f"Fold {fold_idx+1}/{N_FOLDS}: test walks={sorted(test_walks)}, "
          f"test windows={X_test.shape[0]}, accuracy={acc*100:.2f}%")
    fold_accuracies.append(acc)

fold_accuracies = np.array(fold_accuracies)
print()
print(f"Mean accuracy across {N_FOLDS} folds: {fold_accuracies.mean()*100:.2f}%")
print(f"Std deviation: {fold_accuracies.std()*100:.2f}%")
print(f"Min/Max fold accuracy: {fold_accuracies.min()*100:.2f}% / {fold_accuracies.max()*100:.2f}%")