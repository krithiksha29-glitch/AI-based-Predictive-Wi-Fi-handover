import numpy as np
import pandas as pd

# Load what we built
X = np.load('X_data.npy')
y = np.load('y_data.npy')

# We need to know which (set, walk) each window came from, so we split fairly
df = pd.read_csv('pivoted_data.csv')
WINDOW_SIZE = 5

window_set_walk = []
for (set_num, walk_num), group in df.groupby(['set_number', 'walk_number']):
    group = group.reset_index(drop=True)
    n_windows_this_walk = len(group) - WINDOW_SIZE
    if n_windows_this_walk > 0:
        window_set_walk += [(set_num, walk_num)] * n_windows_this_walk

window_set_walk = np.array(window_set_walk)

print("Total windows tracked:", len(window_set_walk))
print("Total windows in X:", X.shape[0])

# Pick which (set, walk) combos go to testing — a fair mix across sets and walk types
test_combos = [(1, 5), (2, 10), (3, 14), (4, 9)]

test_mask = np.array([tuple(sw) in test_combos for sw in window_set_walk])
train_mask = ~test_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("Training windows:", X_train.shape[0])
print("Testing windows:", X_test.shape[0])

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
print("Saved train/test split files")
