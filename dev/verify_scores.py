"""dev/verify_scores.py - deterministic score-verification harness.

Regression guard for the scoring path. Drives it with fixed, hard-coded inputs
and writes the result to JSON. No webcam, no microphone, no model files, no
wall-clock dependence, so two runs are byte-identical.

Run it before and after any edit and diff the two files. Any difference means
the scoring path moved.

    python dev/verify_scores.py --out before.json
    python dev/verify_scores.py --out after.json

Expected values on an unmodified scoring path:
    blink 36.6667 (MILD)   gaze 20.4 (MILD)
    voice 55.0 (MODERATE)  overall 39.9 (MILD)
"""

import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from dementia_analyzer import DementiaAnalyzer
from scoring import (compute_gaze_score, compute_overall_score, explain_scores,
                     risk_level)

# --------------------------------------------------------------------------
# Confounder guard: populate_results() silently swaps the rule-based eye/gaze
# scores for model output if these files exist. Neither is committed. If one
# appears mid-run the comparison would be meaningless, so refuse loudly.
# --------------------------------------------------------------------------
for _m in ("blink_rf_model.pkl", "gaze_rf_model.pkl"):
    if os.path.exists(os.path.join(BASE, _m)):
        raise SystemExit(
            f"ABORT: {_m} is present. It overrides the rule-based score in "
            "gui_app.populate_results() and would invalidate the comparison.")


def _round(obj, nd=6):
    """Recursively round floats so JSON text is byte-stable."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    return obj


# --------------------------------------------------------------------------
# 1. Blink score — deterministic synthetic EAR sequence
# --------------------------------------------------------------------------
def blink_baseline():
    """Drive detect_blink() with explicit timestamps and a stubbed clock.

    Timeline (t in seconds from a fixed origin, 20 Hz):
      0.0 - 5.0   calibration, eyes open (EAR 0.35)
      5.0 - 60.0  a blink every 5 s. Every third blink is held closed for
                  0.7 s, which crosses the 0.5 s micro-sleep threshold.
                  Between blinks, a shallow dip to 0.302 produces partial
                  blinks.

    The dip depth matters. With EAR 0.35 open, calibration fixes
    baseline_ear = 0.350 and ear_threshold = 0.297. A dip must stay ABOVE
    0.297 (or it is a full blink, not a partial one) yet fall below
    threshold + (baseline - threshold) * 0.2 = 0.3076 to be registered.
    0.302 sits in that window.
    """
    a = DementiaAnalyzer()

    T0 = 1_000_000.0          # fixed epoch origin; never time.time()
    DT = 0.05                 # 20 Hz
    N = int(60.0 / DT)

    a.session_start = T0

    blink_starts = [t for t in range(5, 60, 5)]      # 5,10,...,55
    for i in range(N):
        t_rel = i * DT
        ear = 0.35

        for k, bs in enumerate(blink_starts):
            hold = 0.7 if k % 3 == 0 else 0.15       # every 3rd is a micro-sleep
            if bs <= t_rel < bs + hold:
                ear = 0.12
                break
            # shallow dip 1.5 s after each blink -> partial blink
            if bs + 1.5 <= t_rel < bs + 1.7:
                ear = 0.302
                break

        a.detect_blink(ear, ear, timestamp=T0 + t_rel)

    # calculate_blink_rate() and get_session_stats() call time.time(); pin the
    # elapsed window to exactly 60 s so the rate is reproducible.
    import dementia_analyzer as _mod
    real_time_mod = _mod.time

    class _FixedClock:
        @staticmethod
        def time():
            return T0 + 60.0

    _mod.time = _FixedClock
    try:
        risk = a.calculate_dementia_risk()
        stats = a.get_session_stats()
    finally:
        _mod.time = real_time_mod

    return {
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "blink_rate": risk["blink_rate"],
        "blink_variance": risk["blink_variance"],
        "avg_ear": risk["avg_ear"],
        "total_blinks": risk["total_blinks"],
        "score_breakdown": risk["score_breakdown"],
        "risk_factors": risk["risk_factors"],
        "n_micro_sleeps": len(stats["micro_sleeps"]),
        "n_partial_blinks": len(stats["partial_blinks"]),
        "micro_sleep_durations": [round(m["duration"], 4)
                                  for m in stats["micro_sleeps"]],
    }


# --------------------------------------------------------------------------
# 2. Gaze score — fixed dicts
# --------------------------------------------------------------------------
GAZE_CASES = {
    "healthy": {"avg_reaction_time": 0.40, "avg_saccade_speed": 320.0,
                "accuracy": 100.0, "trials": 10},
    "impaired": {"avg_reaction_time": 1.30, "avg_saccade_speed": 80.0,
                 "accuracy": 60.0, "trials": 10},
    "not_run": {"trials": 0},
    # A session-shaped input. Kept fixed so this harness stays a pure
    # scoring-path check: it measures what compute_gaze_score() does with a
    # given input, not what a live session happens to feed in.
    "session_like": {"avg_reaction_time": 0.62, "avg_saccade_speed": 10.4,
                     "accuracy": 100.0, "trials": 10},
}


# --------------------------------------------------------------------------
# 3. Fusion
# --------------------------------------------------------------------------
FUSION_CASES = {
    "all_three": dict(eye_score=30.0, gaze_score=50.0, voice_score=70.0),
    "no_voice": dict(eye_score=30.0, gaze_score=50.0, voice_score=None),
    "eye_only": dict(eye_score=42.0, gaze_score=None, voice_score=None),
    "gaze_voice": dict(eye_score=None, gaze_score=50.0, voice_score=70.0),
    "none": dict(eye_score=None, gaze_score=None, voice_score=None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blink = blink_baseline()
    gaze = {k: compute_gaze_score(v) for k, v in GAZE_CASES.items()}
    fusion = {k: compute_overall_score(**v) for k, v in FUSION_CASES.items()}

    # The end-to-end number a real session would report, built from 1 + 2 with a
    # fixed voice score standing in for the model output.
    VOICE_FIXED = 55.0
    session_overall = compute_overall_score(
        eye_score=blink["risk_score"],
        gaze_score=gaze["session_like"]["score"],
        voice_score=VOICE_FIXED,
    )
    session_explanation = explain_scores(
        eye_result={"risk_score": blink["risk_score"],
                    "score_breakdown": blink["score_breakdown"]},
        gaze_result=gaze["session_like"],
        voice_result={"risk_pct": VOICE_FIXED},
        overall=session_overall,
    )

    payload = {
        "_note": "Deterministic output of dev/verify_scores.py.",
        "blink_score": blink,
        "gaze_scores": gaze,
        "voice_score_fixed_input": VOICE_FIXED,
        "fusion_scores": fusion,
        "session": {
            "eye_score": blink["risk_score"],
            "eye_level": blink["risk_level"],
            "gaze_score": gaze["session_like"]["score"],
            "gaze_level": gaze["session_like"]["level"],
            "voice_score": VOICE_FIXED,
            "voice_level": risk_level(VOICE_FIXED),
            "overall_score": session_overall["score"],
            "overall_level": session_overall["level"],
            "weights": session_overall["weights"],
            "used": session_overall["used"],
        },
        "session_explanation": {
            "overall_score": session_explanation["overall_score"],
            "overall_level": session_explanation["overall_level"],
            "modality_totals": session_explanation["modality_totals"],
            "contributions": [
                {"modality": c["modality"], "feature": c["feature"],
                 "points": c["points"], "contribution": c["contribution"]}
                for c in session_explanation["contributions"]
            ],
        },
        "risk_bands": {str(s): risk_level(s)
                       for s in [0, 19.9, 20, 39.9, 40, 59.9, 60, 79.9, 80, 100]},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_round(payload), f, indent=2, sort_keys=True)
        f.write("\n")

    s = payload["session"]
    print(f"wrote {args.out}")
    print(f"  blink   {s['eye_score']:.4f}  ({s['eye_level']})")
    print(f"  gaze    {s['gaze_score']:.4f}  ({s['gaze_level']})")
    print(f"  voice   {s['voice_score']:.4f}  ({s['voice_level']})")
    print(f"  overall {s['overall_score']:.4f}  ({s['overall_level']})")


if __name__ == "__main__":
    main()
