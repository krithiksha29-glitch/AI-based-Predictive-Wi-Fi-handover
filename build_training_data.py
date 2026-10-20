import pandas as pd
import numpy as np

# Load the pivoted data (one row per moment, all 4 towers + accel side by side)
df = pd.read_csv('pivoted_data.csv')

WINDOW_SIZE = 5
towers = ['Tower_AP1', 'Tower_AP2', 'Tower_AP3', 'Tower_AP4']
accel_cols = ['accel_magnitude']
feature_cols = towers + accel_cols   # 7 features per timestep now, instead of 4

X = []   # will hold the "history" — the input the AI sees (now RSSI + motion)
y = []   # will hold the "answer" — which tower is correct next
for (set_num, walk_num), group in df.groupby(['set_number', 'walk_number']):
    group = group.reset_index(drop=True)

    for i in range(len(group) - WINDOW_SIZE):
        history = group.loc[i:i+WINDOW_SIZE-1, feature_cols].values   # 5 rows x 7 features
        next_row = group.loc[i+WINDOW_SIZE, towers]
        best_next_tower = next_row.idxmax()
        answer = towers.index(best_next_tower)   # turn tower name into a number (0,1,2,3)

        X.append(history)
        y.append(answer)

X = np.array(X)
y = np.array(y)

print("X shape (windows, timesteps, features):", X.shape)
print("y shape (windows,):", y.shape)
print()
print("Example window 0 — history (4 RSSI + 3 accel columns):")
print(X[0])
print("Example window 0 — answer (0=AP1,1=AP2,2=AP3,3=AP4):", y[0])

# Save for the next step
np.save('X_data.npy', X)
np.save('y_data.npy', y)
print()
print("Saved X_data.npy and y_data.npy")