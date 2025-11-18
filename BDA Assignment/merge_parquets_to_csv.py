# merge_parquets_to_csv.py
import pandas as pd
import os
import glob

# Directory containing parquet files
parquet_dir = "data/stream_output/predictions"
output_csv = os.path.join(parquet_dir, "predictions.csv")

# Find all parquet files
parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))

if not parquet_files:
    print("No parquet files found in", parquet_dir)
    exit(1)

# Read all parquet files and concatenate
df_list = [pd.read_parquet(f) for f in parquet_files]
df_all = pd.concat(df_list, ignore_index=True)

# Write to single CSV
df_all.to_csv(output_csv, index=False)
print(f"✅ All {len(parquet_files)} parquet files merged into {output_csv}")
