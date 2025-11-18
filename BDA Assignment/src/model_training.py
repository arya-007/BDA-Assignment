# src/model_training.py

import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def normalize_pid(pid):
    """Normalize Participant IDs like '300_P' -> '300'."""
    return str(pid).split('_')[0]

def train(
    fused_parquet="data/processed/fused/all_features.parquet",
    labels="train_split_depression.csv",
    out_model="data/processed/fused/model.joblib"
):
    # --- Load Data ---
    if not os.path.exists(fused_parquet):
        raise FileNotFoundError("Fused parquet not found: " + fused_parquet)
    if not os.path.exists(labels):
        raise FileNotFoundError("Labels CSV not found: " + labels)

    feats = pd.read_parquet(fused_parquet)
    labs = pd.read_csv(labels)

    # Normalize IDs
    feats['pid_short'] = feats['Participant_ID'].astype(str).apply(normalize_pid)
    labs['pid_short'] = labs['Participant_ID'].astype(str).apply(normalize_pid)

    # Merge features and labels
    merged = feats.merge(
        labs[['pid_short', 'PHQ8_Score', 'PHQ8_Binary']],
        on='pid_short',
        how='inner'
    )

    if merged.empty:
        raise ValueError("No overlapping participants between features and labels. Check naming consistency.")

    # --- Prepare Features and Labels ---
    y = merged['PHQ8_Score'].astype(float)
    X = merged.drop(
        columns=['Participant_ID', 'pid_short', 'PHQ8_Score', 'PHQ8_Binary'],
        errors='ignore'
    )

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # --- Impute and Scale ---
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # --- Train Model ---
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluate ---
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # --- Save Model and Preprocessors ---
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(
        {"model": model, "imputer": imputer, "scaler": scaler, "feature_names": list(X.columns)},
        out_model
    )
    print("✅ Saved trained model to:", out_model)

    return model

if __name__ == "__main__":
    train()
