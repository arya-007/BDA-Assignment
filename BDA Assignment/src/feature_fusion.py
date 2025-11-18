# src/feature_fusion.py
import pandas as pd, os, glob
from utils import list_participants, ensure_dir

def load_if_exists(path):
    return pd.read_csv(path) if path and os.path.exists(path) else None

def fuse(raw_dir="data/raw", processed_dir="data/processed", out_parquet="data/processed/fused/all_features.parquet"):
    parts = list_participants(raw_dir)
    rows=[]
    for p in parts:
        pid = p
        parts_files = {
            "audio": os.path.join(processed_dir,"audio_features", f"{pid}.csv"),
            "audio_ext": os.path.join(processed_dir,"audio_features_ext", f"{pid}.csv"),
            "text": os.path.join(processed_dir,"text_features", f"{pid}.csv"),
            "video": os.path.join(processed_dir,"video_features", f"{pid}.csv")
        }
        dfs = []
        for k,v in parts_files.items():
            df = load_if_exists(v)
            if df is not None:
                # prefix columns to avoid collision
                df = df.add_prefix(f"{k}_")
                dfs.append(df.reset_index(drop=True))
        if not dfs:
            continue
        fused = pd.concat(dfs, axis=1)
        fused.insert(0,"Participant_ID", pid)
        rows.append(fused)
    if not rows:
        print("No participants processed")
        return None
    final = pd.concat(rows, axis=0, ignore_index=True)
    ensure_dir(os.path.dirname(out_parquet))
    final.to_parquet(out_parquet, index=False)
    print("Saved fused parquet to", out_parquet)
    return final

if __name__=="__main__":
    fuse()
