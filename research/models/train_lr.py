"""Train and save the v2 LogisticRegression risk scorer."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/Users/aza/Downloads/zakupki/last")
FEATURES_CSV = PROJECT_ROOT / "workspace" / "eval" / "ml_features.csv"
OUTPUT_PATH = Path(__file__).parent / "lr_model.joblib"

FEATURE_COLUMNS = [
    "has_brand", "brand_count", "has_equivalent", "brands_without_equiv",
    "has_units", "has_ranges", "missing_char_count", "restrictive_count",
    "ktru_mentioned", "beyond_ktru", "has_functional", "has_technical",
    "has_quality", "risk_flag_count", "max_confidence", "mean_confidence",
    "tz_char_count",
]


def main() -> None:
    df = pd.read_csv(FEATURES_CSV, dtype={"notice_id": str})
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy").mean()
    auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc").mean()

    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    in_sample_auc = roc_auc_score(y, proba)

    payload = {
        "model": pipeline,
        "feature_names": FEATURE_COLUMNS,
        "metrics": {
            "cv_accuracy_mean": float(acc),
            "cv_roc_auc_mean": float(auc),
            "in_sample_roc_auc": float(in_sample_auc),
            "n_train": int(len(y)),
            "pos_rate": float(np.mean(y)),
        },
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")
    for k, v in payload["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
