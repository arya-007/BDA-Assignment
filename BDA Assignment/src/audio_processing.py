# src/audio_processing.py
import librosa, numpy as np, pandas as pd, os, sys
from utils import save_df, ensure_dir

def extract_librosa_features(wav_path, sr=None):
    y, sr = librosa.load(wav_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats = {
        "duration": float(librosa.get_duration(y=y, sr=sr)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rmse_mean": float(np.mean(rmse)),
        "spec_cent_mean": float(np.mean(spec_cent)),
        "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
    }
    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    return pd.DataFrame([feats])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed/audio_features")
    args = parser.parse_args()

    pid = args.participant
    folder = os.path.join(args.raw_dir, pid)
    wav_candidates = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav")]
    if not wav_candidates:
        print("No wav for", pid); sys.exit(1)
    wav_path = wav_candidates[0]
    df = extract_librosa_features(wav_path)
    save_df(df, args.out_dir, pid)
    print("Saved audio features for", pid)
