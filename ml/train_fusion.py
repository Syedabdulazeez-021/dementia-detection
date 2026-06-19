#!/usr/bin/env python3
"""
train_fusion.py - Learn the SINGLE combined score from all three tests.

Right now the one overall score is a weighted average with hand-picked weights
(voice 0.40, blink 0.35, gaze 0.25). This script learns that combination from
data instead, producing one model that takes the three per-test scores and
outputs a single risk probability.

REQUIRED DATA - and this is the strict part:
  A CSV where each row is ONE PERSON who did ALL THREE tests, with columns:
      eye_score, gaze_score, voice_score, true_label
  (scores 0-100; true_label 0 = normal, 1 = dementia).

  You cannot build this by merging different datasets - the three scores must
  come from the SAME person. That is what makes it a real multimodal model.
  Until you collect such same-subject data, keep using the built-in weighted
  fusion (which always works); this script is ready for when you have it.

  python train_fusion.py --data multimodal_labelled.csv
"""
import os
import sys
import argparse
import numpy as np

try:
    import pandas as pd
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
except Exception as e:
    print(f"[ERROR] Missing dependency: {e}. Activate your venv first.")
    sys.exit(1)

COLS = ["eye_score", "gaze_score", "voice_score"]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "fusion_model.pkl"))
    ap.add_argument("--min", type=int, default=40)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] No file at {args.data}")
        return
    df = pd.read_csv(args.data)
    missing = [c for c in COLS + ["true_label"] if c not in df.columns]
    if missing:
        print(f"[ERROR] CSV missing columns: {missing}. Expected {COLS + ['true_label']}")
        return

    df = df[pd.to_numeric(df["true_label"], errors="coerce").isin([0, 1])].copy()
    n = len(df)
    if n < args.min:
        print(f"[INFO] Only {n} same-subject rows; need >= {args.min}.")
        print("       Keep collecting people who completed all three tests.")
        return
    X = df[COLS].astype(float).values
    y = df["true_label"].astype(int).values
    if len(np.unique(y)) < 2:
        print("[INFO] Need both classes present.")
        return

    folds = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    # Logistic regression: interpretable - its coefficients ARE the learned weights.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    clf.fit(X, y)
    joblib.dump({"model": clf, "features": COLS}, args.out)

    # Report the learned weights vs the current hand-picked ones.
    w = clf.coef_[0]
    wn = np.abs(w) / (np.abs(w).sum() + 1e-9)
    print("\n===== LEARNED FUSION MODEL =====")
    print(f"  same-subject rows = {n}   CV AUC = {auc.mean():.3f}")
    print("  Learned relative importance (vs current heuristic):")
    for c, val, cur in zip(COLS, wn, [0.35, 0.25, 0.40]):
        print(f"    {c:12s}: learned {val:.2f}   (current {cur:.2f})")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
