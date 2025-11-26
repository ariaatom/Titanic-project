import pandas as pd

# Step 1: Load the original CSV
df = pd.read_csv("train.csv")

# Step 2: Make a working copy (safe to clean)
work_df = df.copy()

# Step 3: Show first 5 rows to understand the data
print(work_df.head())

# Step 4: Check for missing values
print(work_df.isna().sum())
