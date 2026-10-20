import pandas as pd
import numpy as np

# Load your cleaned data
df = pd.read_csv('cleaned_merged_walk_data.csv')

# Round time to the nearest 100ms, so readings from the same scan round merge together
df['time_bucket'] = (df['relative_time_ms'] / 100).round() * 100

# Pivot: one row per (walk, time_bucket), one column per tower's RSSI
pivoted = df.pivot_table(
    index=['set_number', 'walk_number', 'time_bucket'],
    columns='tower',
    values='rssi'
).reset_index()

# Accelerometer readings are the same across all 4 tower rows for a given moment
# (one physical motion reading per time_bucket), so average them back in here
accel = df.groupby(['set_number', 'walk_number', 'time_bucket'])[['accel_x', 'accel_y', 'accel_z']].mean().reset_index()
accel['accel_magnitude'] = np.sqrt(accel['accel_x']**2 + accel['accel_y']**2 + accel['accel_z']**2)
accel = accel[['set_number', 'walk_number', 'time_bucket', 'accel_magnitude']]
pivoted = pivoted.merge(accel, on=['set_number', 'walk_number', 'time_bucket'], how='left')

# Fill small gaps using nearby readings, walk by walk
filled_groups = []
for (set_num, walk_num), group in pivoted.groupby(['set_number', 'walk_number']):
    group = group.interpolate(method='linear', limit_direction='both')
    filled_groups.append(group)

pivoted = pd.concat(filled_groups, ignore_index=True)

# Any remaining RSSI gaps mean a tower was never detected at all during that walk
# (genuinely out of range) — fill with -100 dBm, representing "no signal"
tower_cols = [c for c in pivoted.columns if c.startswith('Tower_')]
pivoted[tower_cols] = pivoted[tower_cols].fillna(-100)

# Any remaining accel gaps (rare, e.g. very start/end of a walk) — fill with 0
pivoted[['accel_magnitude']] = pivoted[['accel_magnitude']].fillna(0)
# Check how many gaps remain after filling
remaining_nan = pivoted.isna().sum().sum()
print(f"Remaining NaN after filling: {remaining_nan}")

pivoted.to_csv('pivoted_data.csv', index=False)
print("Pivoted data saved as pivoted_data.csv")
print(pivoted.head(10))