import pandas as pd
import numpy as np

# Load the pivoted data (one row per moment, all 4 towers + accel magnitude side by side)
df = pd.read_csv('pivoted_data.csv')

WINDOW_SIZE = 5
towers = ['Tower_AP1', 'Tower_AP2', 'Tower_AP3', 'Tower_AP4']
feature_cols = towers + ['accel_magnitude']   # 5 columns: 4 RSSI + 1 accel magnitude (for now)

X = []
y = []
for (set_num, walk_num), group in df.groupby(['set_number', 'walk_number']):
    group = group.reset_index(drop=True)

    for i in range(len(group) - WINDOW_SIZE):
        history = group.loc[i:i+WINDOW_SIZE-1, feature_cols].values.copy()   # 5 rows x 5 cols
        next_row = group.loc[i+WINDOW_SIZE, towers]
        best_next_tower = next_row.idxmax()
        answer = towers.index(best_next_tower)

        # Replace the raw per-timestep accel magnitude column with a single
        # "how much did motion vary across this 5-step window" number,
        # repeated at every timestep so the array shape stays (5, 5).
        accel_magnitudes = history[:, 4]
        accel_variance = np.var(accel_magnitudes)
        history[:, 4] = accel_variance

        X.append(history)
        y.append(answer)

X = np.array(X)
y = np.array(y)

print("X shape (windows, timesteps, features):", X.shape)
print("y shape (windows,):", y.shape)
print()
print("Example window 0 — history (4 RSSI + accel variance, broadcast):")
print(X[0])
print("Example window 0 — answer (0=AP1,1=AP2,2=AP3,3=AP4):", y[0])

np.save('X_data.npy', X)
np.save('y_data.npy', y)
print()
print("Saved X_data.npy and y_data.npy (accel VARIANCE version)")