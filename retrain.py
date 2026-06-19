#!/usr/bin/env python3
"""
retrain.py - Honest feedback-driven retraining for the voice Random Forest.

HOW THE "LEARN FROM MISTAKES" LOOP ACTUALLY WORKS
-------------------------------------------------
A classifier can only learn from CONFIRMED ground-truth labels - never from its
own unverified predictions (training on your own guesses degrades a model). So
the loop here is:

  1. Every voice session is logged to `retrain_data.csv` with its 21 features
     and the model's prediction, plus a blank `true_label` column.
  2. When a real diagnosis is later confirmed for a patient, you open that CSV
     and fill in `true_label`:  0 = cognitively normal,  1 = dementia.
  3. Run this script. It trains a fresh Random Forest on the confirmed rows,
     compares it (with cross-validation) against the current model, and ONLY
     replaces the model if the new one genuinely scores higher. The old model
     is always backed up first.

This will not improve anything until you have accumulated enough confirmed
labels - that is expected and honest. It is the correct mechanism, not a magic
self-improving model.

USAGE:
    python retrain.py                # uses ./retrain_data.csv
    python retrain.py --min 30       # require >= 30 labelled rows
"""
import os
import sys
import shutil
import argparse
from datetime import datetime

import numpy as np

try:
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
except Exception as e:  # pragma: no cover
    print(f"[ERROR] Missing dependency: {e}. Activate your venv first.")
    sys.exit(1)

# The 21 features, in the exact order the model expects.
FEATURE_ORDER = [
    'MFCC2', 'kurt_MFCC30', 'mean_MFCC30', 'skew_MFCC2', 'mean_MFCC16',
    'flt_bnk_eng22', 'MFCC30', 'kurt_MFCC16', 'flt_bnk_eng2', 'flt_bnk_eng24',
    'MFCC1', 'flt_bnk_eng15', 'kurt_MFCC2', 'flt_bnk_eng20', 'flt_bnk_eng13',
    'n_sil_segments', 'frac_silence', 'min_sil_len', 'jitter', 'shimmer', 'HNR',
]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'retrain_data.csv'))
    ap.add_argument('--model', default=os.path.join(HERE, 'dementia_rf_model.pkl'))
    ap.add_argument('--scaler', default=os.path.join(HERE, 'scaler.pkl'))
    ap.add_argument('--min', type=int, default=20,
                    help='minimum confirmed-label rows required (default 20)')
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[INFO] No feedback file yet at {args.data}.")
        print("       Run some voice sessions, confirm diagnoses, fill in the")
        print("       'true_label' column (0/1), then run this again.")
        return

    df = pd.read_csv(args.data)
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        print(f"[ERROR] Feedback file missing feature columns: {missing[:3]}...")
        return

    # Keep only rows the clinician has actually labelled (0 or 1).
    df = df[pd.to_numeric(df.get('true_label'), errors='coerce').isin([0, 1])].copy()
    n = len(df)
    if n < args.min:
        print(f"[INFO] Only {n} confirmed-label rows; need >= {args.min} to "
              f"retrain responsibly.")
        print("       Keep collecting confirmed diagnoses - this is expected.")
        return

    X = df[FEATURE_ORDER].astype(float).values
    y = df['true_label'].astype(int).values
    if len(np.unique(y)) < 2:
        print("[INFO] Need both classes (0 and 1) present to train. Add more.")
        return

    # Scale with the existing scaler so the feature space matches the pipeline.
    try:
        scaler = joblib.load(args.scaler)
        Xs = scaler.transform(X)
    except Exception:
        print("[WARN] Could not load scaler; training on raw features.")
        Xs = X

    folds = min(5, np.bincount(y).min())
    if folds < 2:
        print("[INFO] Too few examples in one class for cross-validation yet.")
        return
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # Score the CURRENT model on the confirmed data (the baseline to beat).
    old_acc = None
    if os.path.exists(args.model):
        try:
            old = joblib.load(args.model)
            old_acc = float(np.mean(old.predict(Xs) == y))
        except Exception:
            old_acc = None

    # Train + cross-validate a fresh model.
    new = RandomForestClassifier(n_estimators=300, random_state=42,
                                 class_weight='balanced')
    cv_acc = float(np.mean(cross_val_score(new, Xs, y, cv=cv, scoring='accuracy')))
    new.fit(Xs, y)

    print("\n==================  RETRAINING REPORT  ==================")
    print(f"  Confirmed-label rows used : {n}  (class balance {np.bincount(y)})")
    print(f"  New model  CV accuracy    : {cv_acc*100:.1f}%")
    if old_acc is not None:
        print(f"  Current model on this data: {old_acc*100:.1f}%")
    print("=========================================================")

    # Only replace if the new model is at least as good as the current one.
    if old_acc is not None and cv_acc < old_acc:
        out = os.path.join(HERE, 'dementia_rf_model_candidate.pkl')
        joblib.dump(new, out)
        print(f"[KEPT OLD] New model did NOT beat the current one. Saved the")
        print(f"           candidate to {out} for inspection; live model untouched.")
        return

    backup = args.model.replace('.pkl', f'_backup_{datetime.now():%Y%m%d_%H%M%S}.pkl')
    if os.path.exists(args.model):
        shutil.copy2(args.model, backup)
        print(f"[BACKUP]   Previous model saved to {os.path.basename(backup)}")
    joblib.dump(new, args.model)
    print(f"[UPDATED]  New model saved to {os.path.basename(args.model)}.")
    print("           Restart the app to use it.")


if __name__ == '__main__':
    main()
