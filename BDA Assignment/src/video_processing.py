# src/video_processing.py
import pandas as pd, numpy as np, os, sys
from utils import save_df

def safe_read(path):
    try:
        return pd.read_csv(path, sep=None, engine='python', encoding_errors='ignore')
    except Exception:
        print(f"⚠️ Skipping non-readable file: {path}")
        return pd.DataFrame()


def summarize_numeric(df):
    if df is None:
        return {}
    ndf = df.select_dtypes(include=[np.number])
    if ndf.empty:
        return {}
    return {
        "cols": int(ndf.shape[1]),
        "rows": int(ndf.shape[0]),
        "mean_all": float(ndf.mean().mean()),
        "std_all": float(ndf.std().mean()),
        "mean_abs_diff": float(ndf.diff().abs().mean().mean())
    }

def extract_video(folder):
    files = {f.lower(): os.path.join(folder,f) for f in os.listdir(folder)}
    def find_key(keys):
        for k in keys:
            for fname in files:
                if k in fname:
                    return files[fname]
        return None
    au = find_key(["au.txt","aus.txt","_au","au."])
    feat2d = find_key(["features.txt","clnf_features.txt","_features.txt"])
    feat3d = find_key(["features3d.txt","3d.txt","features3D".lower()])
    gaze = find_key(["gaze.txt"])
    pose = find_key(["pose.txt"])
    hog = find_key(["hog.txt"])
    dfs = { "au": safe_read(au), "feat2d": safe_read(feat2d), "feat3d": safe_read(feat3d), "gaze": safe_read(gaze), "pose": safe_read(pose), "hog": safe_read(hog)}
    stats = {}
    for k,v in dfs.items():
        s = summarize_numeric(v)
        stats.update({f"{k}_{kk}": vv for kk,vv in s.items() for kk,vv in [(kk, s[kk])] } if s else {})
    return pd.DataFrame([stats])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed/video_features")
    args = parser.parse_args()

    pid=args.participant
    folder=os.path.join(args.raw_dir,pid)
    df = extract_video(folder)
    save_df(df, args.out_dir, pid)
    print("Saved video features for", pid)
