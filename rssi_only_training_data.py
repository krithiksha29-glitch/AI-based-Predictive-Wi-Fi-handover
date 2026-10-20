import pandas as pd
import numpy as np

df = pd.read_csv('pivoted_data.csv')
WINDOW_SIZE = 5
towers = ['Tower_AP1', 'Tower_AP2', 'Tower_AP3', 'Tower_AP4']

X = []
y = []
for (set_num, walk_num), group in df.groupby(['set_number', 'walk_number']):
    group = group.reset_index(drop=True)
    for i in range(len(group) - WINDOW_SIZE):
        history = group.loc[i:i+WINDOW_SIZE-1, towers].values
        next_row = group.loc[i+WINDOW_SIZE, towers]
        best_next_tower = next_row.idxmax()
        answer = towers.index(best_next_tower)
        X.append(history)
        y.append(answer)

X = np.array(X)
y = np.array(y)
print("RSSI-only X shape:", X.shape)
np.save('X_data.npy', X)
np.save('y_data.npy', y)
print("Overwrote X_data.npy and y_data.npy with RSSI-only (4 feature) version")