import pandas as pd

# Load the pivoted data
df = pd.read_csv('pivoted_data.csv')

WINDOW_SIZE = 5
towers = ['Tower_AP1', 'Tower_AP2', 'Tower_AP3', 'Tower_AP4']

rows = []

# Go through each walk separately — we never mix data across different walks
for (set_num, walk_num), group in df.groupby(['set_number', 'walk_number']):
    group = group.reset_index(drop=True)
    
    # Slide a window of 5 across this walk's readings
    for i in range(len(group) - WINDOW_SIZE):
        history = group.loc[i:i+WINDOW_SIZE-1, towers]   # 5 rows of history
        next_row = group.loc[i+WINDOW_SIZE, towers]        # the row right after

        # Which tower is strongest in the next row? (that's our "answer")
        best_next_tower = next_row.idxmax()

        # Which tower is strongest right now (last row of history)?
        current_best_tower = history.iloc[-1].idxmax()

        # Is a handover about to happen? (Option B)
        handover = best_next_tower != current_best_tower

        rows.append({
	'set_number':set_num,
            'walk_number': walk_num,
            'window_start_index': i,
            'current_best_tower': current_best_tower,
            'next_best_tower': best_next_tower,
            'handover': handover
        })

result = pd.DataFrame(rows)
result.to_csv('windows_summary.csv', index=False)
print("Saved windows_summary.csv")
print(result.head(15))
print()
print("Total windows created:", len(result))
print("Handover windows:", result['handover'].sum())