# src/utils.py
import os, glob, pandas as pd

def list_participants(raw_dir="data/raw"):
    parts = [os.path.basename(p) for p in glob.glob(os.path.join(raw_dir, "*")) if os.path.isdir(p)]
    return sorted(parts)

def ensure_dir(d):
    import os
    os.makedirs(d, exist_ok=True)

def short_id_from_folder(folder_name):
    return str(folder_name).split('_')[0]

def save_df(df, out_dir, pid):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{pid}.csv")
    df.to_csv(path, index=False)
    return path
