import numpy as np
import pandas as pd

X_train = np.load('X_train.npy')
print("Any NaN in X_train?", np.isnan(X_train).any())
print("Total NaN count:", np.isnan(X_train).sum())
print("Total values:", X_train.size)

# Check the pivoted data directly too
df = pd.read_csv('pivoted_data.csv')
print()
print("NaN in pivoted_data.csv:")
print(df.isna().sum())