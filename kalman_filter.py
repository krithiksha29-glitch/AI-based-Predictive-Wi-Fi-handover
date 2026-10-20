import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your cleaned data (before pivoting)
df = pd.read_csv('cleaned_merged_walk_data.csv')

def kalman_filter_1d(readings, process_variance=5.0, measurement_variance=10.0):
    """
    A simple 1D Kalman filter.
    process_variance: how much we expect the TRUE signal to naturally change 
                       between readings (small = we expect smooth, gradual change)
    measurement_variance: how noisy/unreliable we think our sensor readings are 
                           (larger = we trust the sensor less, smooth more)
    """
    n = len(readings)
    estimates = np.zeros(n)
    error_estimate = 1.0

    estimates[0] = readings[0]  # start with the first real reading

    for i in range(1, n):
        # Predict step
        predicted_estimate = estimates[i-1]
        predicted_error = error_estimate + process_variance

        # Update/Correct step
        kalman_gain = predicted_error / (predicted_error + measurement_variance)
        estimates[i] = predicted_estimate + kalman_gain * (readings[i] - predicted_estimate)
        error_estimate = (1 - kalman_gain) * predicted_error

    return estimates

# Apply the filter separately for each (set, walk, tower) combination
df = df.sort_values(['set_number', 'walk_number', 'tower', 'timestamp']).reset_index(drop=True)

filtered_rssi = []
for (set_num, walk_num, tower), group in df.groupby(['set_number', 'walk_number', 'tower']):
    raw = group['rssi'].values
    smoothed = kalman_filter_1d(raw)
    filtered_rssi.append(pd.Series(smoothed, index=group.index))

df['rssi_kalman'] = pd.concat(filtered_rssi).sort_index()

df.to_csv('cleaned_merged_walk_data_kalman.csv', index=False)
print("Saved cleaned_merged_walk_data_kalman.csv with new 'rssi_kalman' column")

# Plot a comparison: raw vs Kalman-filtered, for one example walk
example = df[(df.set_number==1) & (df.walk_number==1) & (df.tower=='Tower_AP1')]

plt.figure(figsize=(10,6))
plt.plot(example['timestamp'], example['rssi'], marker='o', alpha=0.5, label='Raw RSSI')
plt.plot(example['timestamp'], example['rssi_kalman'], marker='o', label='Kalman Filtered')
plt.title("Set 1, Walk 1, AP1 — Raw vs Kalman Filtered RSSI")
plt.xlabel("Time (ms)")
plt.ylabel("RSSI (dBm)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('kalman_comparison.png')
print("Saved kalman_comparison.png")