# src/audio_features_ext.py
import pandas as pd, numpy as np, os, sys
from utils import save_df, ensure_dir

def summarize_numeric(df):
    dfn = df.select_dtypes(include=[np.number])
    if dfn.empty:
        return {}
    return {
        "mean_all": float(dfn.mean().mean()),
        "std_all": float(dfn.std().mean()),
        "cols": int(dfn.shape[1]),
        "rows": int(dfn.shape[0])
    }

def extract_covarep_formant(covarep_path, formant_path):
    out = {}
    if covarep_path and os.path.exists(covarep_path):
        try:
            df = pd.read_csv(covarep_path)
            s = summarize_numeric(df)
            out.update({f"covarep_{k}": v for k,v in s.items()})
            if "F0" in df.columns:
                out["covarep_F0_mean"] = float(df["F0"].dropna().mean())
                out["covarep_F0_std"] = float(df["F0"].dropna().std())
        except Exception as e:
            out["covarep_error"] = str(e)
    if formant_path and os.path.exists(formant_path):
        try:
            df2 = pd.read_csv(formant_path)
            s2 = summarize_numeric(df2)
            out.update({f"formant_{k}": v for k,v in s2.items()})
        except Exception as e:
            out["formant_error"] = str(e)
    return pd.DataFrame([out])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed/audio_features_ext")
    args = parser.parse_args()

    pid=args.participant
    folder=os.path.join(args.raw_dir,pid)
    cov=os.path.join(folder, f"{pid.split('_')[0]}_COVAREP.csv")
    form=os.path.join(folder, f"{pid.split('_')[0]}_FORMANT.csv")
    # fallback if names differ
    if not os.path.exists(cov):
        covs=[os.path.join(folder,f) for f in os.listdir(folder) if "covarep" in f.lower()]
        cov=covs[0] if covs else None
    if not os.path.exists(form):
        forms=[os.path.join(folder,f) for f in os.listdir(folder) if "formant" in f.lower()]
        form=forms[0] if forms else None

    df = extract_covarep_formant(cov, form)
    save_df(df, args.out_dir, pid)
    print("Saved extended audio features for", pid)
