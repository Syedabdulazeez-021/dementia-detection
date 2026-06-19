#!/usr/bin/env python3
"""
train_models.py - Train a real ML model for ONE modality from labelled data.

This turns a rule-based channel (blink or gaze) into a trained model, exactly
like the voice channel. It needs a CSV of REAL labelled examples: one row per
person, the modality's features, and a `true_label` column (0 = cognitively
normal, 1 = dementia).

  python train_models.py --modality blink --data blink_labelled.csv
  python train_models.py --modality gaze  --data gaze_labelled.csv
  python train_models.py --modality voice --data voice_labelled.csv

IMPORTANT (read this):
  * Use REAL labelled data only. Each modality must be trained on data where
    the label is a CONFIRMED diagnosis. See ml/README.md for how to obtain it.
  * Do NOT mix people across modalities into one file - that is fabrication.
    A true single combined model needs the SAME people measured on all three
    (see train_fusion.py).
"""
import os
import sys
import argparse
import numpy as np

try:
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
except Exception as e:
    print(f"[ERROR] Missing dependency: {e}. Activate your venv first.")
    sys.exit(1)

# Feature columns expected for each modality (must match these headers).
FEATURES = {
    "blink": ["blink_rate", "blink_variance", "eye_openness",
              "micro_sleeps", "partial_blinks"],
    "gaze":  ["avg_reaction_time", "avg_saccade_speed", "accuracy"],
    "voice": ['MFCC2', 'kurt_MFCC30', 'mean_MFCC30', 'skew_MFCC2', 'mean_MFCC16',
              'flt_bnk_eng22', 'MFCC30', 'kurt_MFCC16', 'flt_bnk_eng2',
              'flt_bnk_eng24', 'MFCC1', 'flt_bnk_eng15', 'kurt_MFCC2',
              'flt_bnk_eng20', 'flt_bnk_eng13', 'n_sil_segments', 'frac_silence',
              'min_sil_len', 'jitter', 'shimmer', 'HNR'],
}
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=list(FEATURES))
    ap.add_argument("--data", required=True, help="labelled CSV path")
    ap.add_argument("--out", default=None, help="output model .pkl path")
    ap.add_argument("--min", type=int, default=30,
                    help="minimum labelled rows required (default 30)")
    args = ap.parse_args()

    cols = FEATURES[args.modality]
    out = args.out or os.path.join(HERE, f"{args.modality}_rf_model.pkl")

    if not os.path.exists(args.data):
        print(f"[ERROR] No file at {args.data}")
        return
    df = pd.read_csv(args.data)

    missing = [c for c in cols + ["true_label"] if c not in df.columns]
    if missing:
        print(f"[ERROR] CSV missing required columns: {missing}")
        print(f"        Expected: {cols + ['true_label']}")
        return

    df = df[pd.to_numeric(df["true_label"], errors="coerce").isin([0, 1])].copy()
    n = len(df)
    if n < args.min:
        print(f"[INFO] Only {n} labelled rows; need >= {args.min}. Collect more.")
        return
    X = df[cols].astype(float).values
    y = df["true_label"].astype(int).values
    if len(np.unique(y)) < 2:
        print("[INFO] Need both classes (0 and 1) present.")
        return

    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        print("[INFO] Too few examples in one class for cross-validation.")
        return
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                 class_weight="balanced")
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1")
    clf.fit(X, y)
    joblib.dump({"model": clf, "features": cols, "modality": args.modality}, out)

    print(f"\n===== {args.modality.upper()} MODEL TRAINED =====")
    print(f"  rows={n}  class balance={np.bincount(y).tolist()}")
    print(f"  CV accuracy = {acc.mean()*100:.1f}% (+/- {acc.std()*100:.1f})")
    print(f"  CV F1       = {f1.mean():.3f}")
    print(f"  saved -> {out}")
    print("  (Drop this .pkl into the project root to let the app use it.)")


if __name__ == "__main__":
    main()
