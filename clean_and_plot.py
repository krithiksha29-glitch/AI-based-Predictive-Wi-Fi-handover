import pandas as pd
import matplotlib.pyplot as plt

# Step A: Load your merged file
df = pd.read_csv('cleaned_merged_walk_data.csv')

# Step B: Sort rows properly (by walk, then by time)
df = df.sort_values(['walk_number', 'timestamp']).reset_index(drop=True)

# Step C: Add relative_time_ms — each walk now starts counting from 0
df['relative_time_ms'] = df.groupby('walk_number')['timestamp'].transform(lambda x: x - x.min())

# Step D: Save the cleaned file
df.to_csv('cleaned_merged_walk_data.csv', index=False)
print("Cleaned file saved as cleaned_merged_walk_data.csv")

# Step E: Plot Walk 1 as an example
walk1 = df[df.walk_number == 1]
plt.figure(figsize=(10, 6))
for tower in sorted(walk1.tower.unique()):
    t = walk1[walk1.tower == tower]
    plt.plot(t.relative_time_ms / 1000, t.rssi, marker='o', label=tower)

plt.title("Walk 1: RSSI vs Time")
plt.xlabel("Time (seconds)")
plt.ylabel("RSSI (dBm)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('walk1_plot.png')
plt.show()
print("Graph saved as walk1_plot.png")