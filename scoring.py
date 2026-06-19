"""
scoring.py — Unified risk scoring for the BIO Dementia Detection System
========================================================================

This module turns the raw measurements collected by each test
(eye/blink, gaze, voice) into a single, comparable 0-100 risk score, and
then fuses the three modalities into one overall prediction.

IMPORTANT — these are *heuristic screening* scores, not a clinical
diagnosis. The thresholds below are based on commonly reported research
ranges (blink rate, saccadic reaction time, etc.) and are intentionally
conservative. They are documented inline so they can be tuned easily.

Every score follows the same convention:
    0   = no risk indicators / healthy range
    100 = strong abnormality across all measured factors

Risk bands (shared by all modalities and the overall score):
    0-20   LOW
    20-40  MILD
    40-60  MODERATE
    60-80  HIGH
    80-100 VERY HIGH
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clamp(value, low=0.0, high=100.0):
    """Constrain a value to the [low, high] range."""
    return max(low, min(high, value))


def _lerp_risk(value, healthy_at, max_risk_at, max_points):
    """
    Linearly map a measurement to risk points.

    - At `healthy_at` (and anything more healthy) the risk is 0.
    - At `max_risk_at` (and beyond) the risk is `max_points`.
    - In between it scales linearly.

    Works whether higher values are worse (healthy_at < max_risk_at) or
    lower values are worse (healthy_at > max_risk_at).
    """
    if healthy_at == max_risk_at:
        return 0.0
    frac = (value - healthy_at) / (max_risk_at - healthy_at)
    return _clamp(frac, 0.0, 1.0) * max_points


def risk_level(score):
    """Convert a 0-100 score into a categorical band."""
    if score < 20:
        return "LOW"
    elif score < 40:
        return "MILD"
    elif score < 60:
        return "MODERATE"
    elif score < 80:
        return "HIGH"
    else:
        return "VERY HIGH"


# ---------------------------------------------------------------------------
# Gaze score
# ---------------------------------------------------------------------------
#
# The gaze stimulus experiment measures how quickly and accurately the eyes
# move to a peripheral target. Cognitive decline is associated with:
#   • slower saccadic reaction time
#   • reduced accuracy (more missed / wrong-direction responses)
#   • slower / hypometric saccades
#
# Weights (sum to 100):
#   Reaction time .... 45   (most informative, well-studied)
#   Accuracy ......... 40
#   Saccade speed .... 15   (kept low: px/s is highly camera/-distance
#                            dependent, so it is a weak absolute signal)
# ---------------------------------------------------------------------------

# Reaction time (seconds): includes webcam + processing latency, so the
# "healthy" anchor is higher than the ~0.2 s of a pure saccade.
GAZE_RT_HEALTHY_S = 0.5
GAZE_RT_MAX_RISK_S = 1.5

# Accuracy (%): perfect is 0 risk, 50% or below is treated as maximal.
GAZE_ACC_HEALTHY = 100.0
GAZE_ACC_MAX_RISK = 50.0

# Saccade speed (pixels/second): faster is healthier.
GAZE_SPEED_HEALTHY = 300.0
GAZE_SPEED_MAX_RISK = 50.0


def compute_gaze_score(gaze_results):
    """
    Compute a 0-100 gaze risk score from the gaze experiment results.

    Args:
        gaze_results: dict with keys
            avg_reaction_time (s), avg_saccade_speed (px/s),
            accuracy (%), trials (int)

    Returns:
        dict with:
            available     bool   False if no trials were completed
            score         float  0-100 risk score (0 if unavailable)
            level         str    risk band
            factors       list   human-readable abnormality strings
            breakdown     dict   points contributed by each factor
    """
    trials = int(gaze_results.get("trials", 0) or 0)
    if trials <= 0:
        return {
            "available": False,
            "score": 0.0,
            "level": "N/A",
            "factors": ["Gaze test not completed"],
            "breakdown": {},
        }

    rt = float(gaze_results.get("avg_reaction_time", 0.0) or 0.0)
    speed = float(gaze_results.get("avg_saccade_speed", 0.0) or 0.0)
    acc = float(gaze_results.get("accuracy", 0.0) or 0.0)

    factors = []

    # Reaction time (higher = worse)
    rt_pts = _lerp_risk(rt, GAZE_RT_HEALTHY_S, GAZE_RT_MAX_RISK_S, 45)
    if rt > GAZE_RT_HEALTHY_S:
        factors.append(f"Slow gaze reaction time ({rt:.2f}s)")

    # Accuracy (lower = worse)
    acc_pts = _lerp_risk(acc, GAZE_ACC_HEALTHY, GAZE_ACC_MAX_RISK, 40)
    if acc < 90.0:
        factors.append(f"Reduced gaze accuracy ({acc:.0f}%)")

    # Saccade speed (lower = worse). Only score if we actually measured it.
    speed_pts = 0.0
    if speed > 0:
        speed_pts = _lerp_risk(speed, GAZE_SPEED_HEALTHY, GAZE_SPEED_MAX_RISK, 15)
        if speed < GAZE_SPEED_HEALTHY:
            factors.append(f"Slow saccade speed ({speed:.0f} px/s)")

    score = _clamp(rt_pts + acc_pts + speed_pts)

    return {
        "available": True,
        "score": round(score, 1),
        "level": risk_level(score),
        "factors": factors,
        "breakdown": {
            "reaction_time": round(rt_pts, 1),
            "accuracy": round(acc_pts, 1),
            "saccade_speed": round(speed_pts, 1),
        },
    }


# ---------------------------------------------------------------------------
# Overall fused score
# ---------------------------------------------------------------------------
#
# The three modalities are combined into one prediction using a weighted
# average. The voice model carries the most weight because it is a trained
# machine-learning classifier (vs. the rule-based eye/gaze heuristics).
#
# Base weights (only over the modalities that are actually available):
#   Voice ...... 0.40
#   Eye/Blink .. 0.35
#   Gaze ....... 0.25
#
# If a modality is missing, its weight is dropped and the remaining
# weights are renormalised so they still sum to 1.0.
# ---------------------------------------------------------------------------

BASE_WEIGHTS = {"voice": 0.40, "eye": 0.35, "gaze": 0.25}


def compute_overall_score(eye_score=None, gaze_score=None, voice_score=None):
    """
    Fuse the available modality scores into one overall risk prediction.

    Args:
        eye_score:   float 0-100 or None
        gaze_score:  float 0-100 or None
        voice_score: float 0-100 or None  (voice risk_pct)

    Returns:
        dict with:
            available    bool
            score        float 0-100 (0 if nothing available)
            level        str   risk band
            weights      dict  the renormalised weight actually used per modality
            used         list  modalities that contributed
    """
    inputs = {"eye": eye_score, "gaze": gaze_score, "voice": voice_score}

    # Keep only modalities that supplied a usable numeric score.
    present = {k: float(v) for k, v in inputs.items() if v is not None}

    if not present:
        return {
            "available": False,
            "score": 0.0,
            "level": "N/A",
            "weights": {},
            "used": [],
        }

    # Renormalise the base weights over the present modalities.
    total_w = sum(BASE_WEIGHTS[k] for k in present)
    weights = {k: BASE_WEIGHTS[k] / total_w for k in present}

    score = sum(present[k] * weights[k] for k in present)
    score = _clamp(score)

    return {
        "available": True,
        "score": round(score, 1),
        "level": risk_level(score),
        "weights": {k: round(w, 3) for k, w in weights.items()},
        "used": sorted(present.keys()),
    }


# ---------------------------------------------------------------------------
# Explainability (XAI)
# ---------------------------------------------------------------------------
#
# Breaks the final overall score down into the exact contribution of every
# feature, so the result is transparent rather than a black box. The key
# identity is:
#
#     overall_score  =  SUM over modalities m of  weight[m] * score[m]
#                    =  SUM over every feature f of  weight[m(f)] * points[f]
#
# i.e. each feature's contribution to the FINAL score is simply its points
# within its own test, scaled by that test's fusion weight. All the
# contributions therefore add up exactly to the overall score.
# ---------------------------------------------------------------------------

# Human-readable labels + one-line meanings for each raw feature.
_FEATURE_INFO = {
    # eye / blink
    "blink_rate":      ("Blink rate", "How far blinks/minute sit outside the normal 12\u201325 range."),
    "blink_variance":  ("Blink irregularity", "How erratic the gaps between blinks are."),
    "eye_openness":    ("Eye openness", "Reduced average eye-opening (low EAR)."),
    "micro_sleeps":    ("Micro-sleeps", "Eyes held closed >0.5 s \u2014 an attention/drowsiness sign."),
    "partial_blinks":  ("Partial blinks", "Incomplete blinks \u2014 reduced blink amplitude/control."),
    "sleepy_state_bonus": ("Sleepy state (CNN)", "Model-detected sleepy expression (web build only)."),
    # gaze
    "reaction_time":   ("Gaze reaction time", "Slower response to the visual target."),
    "accuracy":        ("Gaze accuracy", "Wrong-direction / missed responses."),
    "saccade_speed":   ("Saccade speed", "Slower eye movements between targets."),
    # voice (single combined contribution)
    "voice_overall":   ("Voice risk (RF model)", "Trained Random-Forest probability of dementia from speech."),
    "blink_ml_model":  ("Blink risk (ML model)", "Trained model probability from blink features."),
    "gaze_ml_model":   ("Gaze risk (ML model)", "Trained model probability from gaze features."),
}

_MODALITY_LABEL = {"eye": "Eye / Blink", "gaze": "Gaze", "voice": "Voice"}


def explain_scores(eye_result=None, gaze_result=None, voice_result=None, overall=None):
    """
    Produce a transparent, feature-level explanation of the overall score.

    Args:
        eye_result:   dict from DementiaAnalyzer.calculate_dementia_risk()
                      (needs 'risk_score' and 'score_breakdown'); or None.
        gaze_result:  dict from compute_gaze_score(); or None.
        voice_result: the voice_results dict (needs 'risk_pct'); or None.
        overall:      dict from compute_overall_score(); recomputed if None.

    Returns:
        dict with:
            overall_score, overall_level
            contributions : list of dicts, each
                {modality, feature, label, meaning, points (within test),
                 weight, contribution (points toward the FINAL score)}
                sorted by contribution descending.
            modality_totals : {modality: contribution_to_overall}
            summary : short plain-language sentence.
    """
    eye_score = eye_result.get("risk_score") if eye_result else None
    gaze_score = gaze_result.get("score") if (gaze_result and gaze_result.get("available")) else None
    voice_score = None
    if voice_result:
        vp = voice_result.get("risk_pct")
        if isinstance(vp, (int, float)):
            voice_score = float(vp)

    if overall is None:
        overall = compute_overall_score(eye_score=eye_score, gaze_score=gaze_score, voice_score=voice_score)

    weights = overall.get("weights", {})
    contributions = []
    modality_totals = {}

    # Eye features
    if eye_score is not None and "eye" in weights:
        w = weights["eye"]
        bd = (eye_result or {}).get("score_breakdown", {}) or {}
        for key, pts in bd.items():
            label, meaning = _FEATURE_INFO.get(key, (key, ""))
            contributions.append({
                "modality": "eye", "feature": key, "label": label, "meaning": meaning,
                "points": round(float(pts), 1), "weight": w,
                "contribution": round(float(pts) * w, 1),
            })
        modality_totals["eye"] = round(eye_score * w, 1)

    # Gaze features
    if gaze_score is not None and "gaze" in weights:
        w = weights["gaze"]
        bd = (gaze_result or {}).get("breakdown", {}) or {}
        for key, pts in bd.items():
            label, meaning = _FEATURE_INFO.get(key, (key, ""))
            contributions.append({
                "modality": "gaze", "feature": key, "label": label, "meaning": meaning,
                "points": round(float(pts), 1), "weight": w,
                "contribution": round(float(pts) * w, 1),
            })
        modality_totals["gaze"] = round(gaze_score * w, 1)

    # Voice. If per-feature attribution is available, split the voice
    # contribution across its top drivers; otherwise show one combined bar.
    if voice_score is not None and "voice" in weights:
        w = weights["voice"]
        voice_total = voice_score * w
        feats = (voice_result or {}).get("feature_contributions", []) or []
        if feats:
            for f in feats:
                frac = float(f.get("fraction", 0))
                contributions.append({
                    "modality": "voice", "feature": f.get("name", "voice"),
                    "label": f.get("label", "Voice feature"),
                    "meaning": "Speech feature flagged by the Random-Forest model.",
                    "points": round(voice_score * frac, 1), "weight": w,
                    "contribution": round(voice_total * frac, 1),
                })
        else:
            label, meaning = _FEATURE_INFO["voice_overall"]
            contributions.append({
                "modality": "voice", "feature": "voice_overall", "label": label, "meaning": meaning,
                "points": round(voice_score, 1), "weight": w,
                "contribution": round(voice_total, 1),
            })
        modality_totals["voice"] = round(voice_total, 1)

    # Sort by contribution (largest driver first)
    contributions.sort(key=lambda c: c["contribution"], reverse=True)

    # Plain-language summary naming the top driver(s)
    drivers = [c for c in contributions if c["contribution"] >= 1.0][:3]
    if drivers:
        names = ", ".join(d["label"].lower() for d in drivers)
        summary = (f"The {overall.get('level','N/A')} overall score "
                   f"({overall.get('score',0):.0f}%) is driven mainly by: {names}.")
    else:
        summary = (f"All measured features are in normal ranges \u2014 overall score "
                   f"{overall.get('score',0):.0f}% ({overall.get('level','N/A')}).")

    return {
        "overall_score": overall.get("score", 0.0),
        "overall_level": overall.get("level", "N/A"),
        "contributions": contributions,
        "modality_totals": modality_totals,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Optional ML models for blink / gaze (rule-based fallback if absent)
# ---------------------------------------------------------------------------
#
# If a trained model file (e.g. blink_rf_model.pkl) is present in the project
# directory, the app uses it to score that modality; otherwise it falls back to
# the rule-based score with no change in behaviour. Models are produced by
# ml/train_models.py from REAL labelled data.

def load_modality_model(modality, base_dir):
    """Return a loaded model bundle {'model','features',...} or None if absent."""
    import os
    path = os.path.join(base_dir, f"{modality}_rf_model.pkl")
    if not os.path.exists(path):
        return None
    try:
        import joblib
        bundle = joblib.load(path)
        # accept either a bare estimator or our {'model','features'} dict
        if isinstance(bundle, dict) and "model" in bundle:
            return bundle
        return {"model": bundle, "features": None}
    except Exception:
        return None


def ml_modality_score(bundle, feature_dict):
    """
    Return a 0-100 risk score = P(dementia)*100 from a loaded model bundle and a
    {feature_name: value} dict, or None on any problem (so caller falls back).
    """
    if not bundle:
        return None
    try:
        feats = bundle.get("features")
        if feats:
            X = [[float(feature_dict[f]) for f in feats]]
        else:
            X = [list(feature_dict.values())]
        proba = bundle["model"].predict_proba(X)[0]
        return float(proba[1]) * 100.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Gaze (healthy):",
          compute_gaze_score({"avg_reaction_time": 0.4, "avg_saccade_speed": 320,
                              "accuracy": 100, "trials": 10}))
    print("Gaze (impaired):",
          compute_gaze_score({"avg_reaction_time": 1.3, "avg_saccade_speed": 80,
                              "accuracy": 60, "trials": 10}))
    print("Gaze (none):", compute_gaze_score({"trials": 0}))
    print("Overall (all):",
          compute_overall_score(eye_score=30, gaze_score=50, voice_score=70))
    print("Overall (no voice):",
          compute_overall_score(eye_score=30, gaze_score=50, voice_score=None))
    print("Overall (eye only):",
          compute_overall_score(eye_score=42, gaze_score=None, voice_score=None))
