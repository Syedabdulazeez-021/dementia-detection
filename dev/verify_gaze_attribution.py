"""dev/verify_gaze_attribution.py - isolate what each gaze fix changes.

`dev/verify_scores.py` feeds compute_gaze_score() a FIXED dict, so it proves the
scoring path itself has not moved. It cannot show what a live session feeds INTO
that path.

This script replays one identical deterministic participant through the gaze
STIMULUS branch with each of the two behavioural fixes toggled independently:

    trial-advance fix   a trial completes on any classified response, plus a
                        per-trial timeout
    iris-reset fix      prev_eye_x is cleared at stimulus onset

The both-on run is cross-checked against the real
`DementiaDetectionGUI._complete_gaze_trial()` so this simulator cannot drift
from production code.

    python dev/verify_gaze_attribution.py
"""

import json
import os
import sys

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from scoring import compute_gaze_score, compute_overall_score

FPS = 30.0
DT = 1.0 / FPS
FIXATION_GAP_S = 2.0          # fixed, not random, for determinism
TIMEOUT_S = 3.0               # mirrors gui_app.GAZE_TRIAL_TIMEOUT_S
CENTER_X = 320.0
TARGET_X = {"LEFT": 250.0, "RIGHT": 390.0}
SACCADE_S = 0.10              # time for the iris to travel to the target

# One deterministic participant. Per trial:
#   target, behaviour, t_first (s from onset), t_correct (s) | None
#   behaviour: "correct" | "wrong_then_correct" | "no_response"
SCRIPT = [
    ("LEFT",  "correct",            0.45, None),
    ("RIGHT", "correct",            0.52, None),
    ("LEFT",  "wrong_then_correct", 0.50, 1.70),   # looks away, then corrects
    ("RIGHT", "correct",            0.48, None),
    ("LEFT",  "no_response",        None, None),   # never looks
    ("RIGHT", "wrong_then_correct", 0.55, 1.85),   # looks away, then corrects
    ("LEFT",  "correct",            0.57, None),
    ("RIGHT", "correct",            0.50, None),
    ("LEFT",  "correct",            0.46, None),
    ("RIGHT", "correct",            0.60, None),
]
OLD_STALL_CAP_S = 12.0        # how long the OLD loop is allowed to spin


def opposite(side):
    return "LEFT" if side == "RIGHT" else "RIGHT"


def frames_for_trial(target, behaviour, t_first, t_correct, max_s):
    """Yield (t_since_onset, gaze_dir, eye_x) at FPS for one trial."""
    out = []
    n = int(max_s / DT) + 1
    for i in range(n):
        t = i * DT
        gaze_dir, aim = "CENTER", CENTER_X
        if behaviour == "correct":
            if t >= t_first:
                gaze_dir, aim = target, TARGET_X[target]
        elif behaviour == "wrong_then_correct":
            if t >= t_correct:
                gaze_dir, aim = target, TARGET_X[target]
            elif t >= t_first:
                w = opposite(target)
                gaze_dir, aim = w, TARGET_X[w]
        # "no_response": stays CENTER

        # iris glides to `aim` over SACCADE_S once the look begins
        onset_t = None
        if behaviour == "correct" and t >= t_first:
            onset_t = t_first
        elif behaviour == "wrong_then_correct":
            if t >= t_correct:
                onset_t = t_correct
            elif t >= t_first:
                onset_t = t_first
        if onset_t is None:
            eye_x = CENTER_X
        else:
            frac = min(1.0, (t - onset_t) / SACCADE_S)
            prev_aim = CENTER_X
            if behaviour == "wrong_then_correct" and t >= t_correct:
                prev_aim = TARGET_X[opposite(target)]
            eye_x = prev_aim + (aim - prev_aim) * frac
        out.append((t, gaze_dir, eye_x))
    return out


def run(fix1, fix2):
    """Replay the whole session through the STIMULUS branch with the two fixes
    toggled. Mirrors gui_app.gaze_capture_loop()'s STIMULUS branch."""
    saccade_speeds, reaction_times, v_peaks, trials = [], [], [], []
    correct_trials = trial_count = 0
    prev_eye_x = prev_time = None
    clock = 0.0
    stalled = []

    for idx, (target, behaviour, t_first, t_correct) in enumerate(SCRIPT, start=1):
        clock += FIXATION_GAP_S                      # fixation gap
        stimulus_time = clock
        trial_speed_start = len(saccade_speeds)

        if fix2:
            # Fix 2: drop the previous trial's iris position at stimulus onset.
            prev_eye_x = prev_time = None

        max_s = TIMEOUT_S if fix1 else OLD_STALL_CAP_S
        response, reaction, done = None, None, False

        for (t, gaze_dir, eye_x) in frames_for_trial(target, behaviour, t_first,
                                                     t_correct, max_s):
            now = stimulus_time + t
            if prev_eye_x is not None:
                dist = abs(eye_x - prev_eye_x)          # unchanged formula
                dtv = now - prev_time
                if dtv > 0:
                    saccade_speeds.append(dist / dtv)   # unchanged formula
            prev_eye_x, prev_time = eye_x, now

            if fix1:
                if gaze_dir in ("LEFT", "RIGHT"):
                    response, reaction, done = gaze_dir, t, True
            else:
                # OLD rule: only an exact match completes a trial.
                if gaze_dir == target:
                    response, reaction, done = gaze_dir, t, True
            if done:
                clock = now
                break

        if not done:
            if fix1:                                   # timeout completes it
                clock = stimulus_time + TIMEOUT_S
                response, reaction = None, None
            else:
                # OLD behaviour: nothing completes the trial. The real loop
                # spins here forever with no timeout.
                clock = stimulus_time + OLD_STALL_CAP_S
                stalled.append(idx)
                continue

        if reaction is not None:
            reaction_times.append(reaction)
        trial_count += 1
        if response is not None and response == target:
            correct_trials += 1
        tsl = saccade_speeds[trial_speed_start:]
        vp = max(tsl) if tsl else 0.0
        v_peaks.append(vp)
        trials.append({"target": target, "response": response,
                       "correct": response == target, "v_peak": vp})

    res = {
        "reaction_times": reaction_times,
        "saccade_speeds": saccade_speeds,
        "v_peaks": v_peaks,
        "avg_reaction_time": (sum(reaction_times) / len(reaction_times)
                              if reaction_times else 0.0),
        "avg_saccade_speed": (sum(saccade_speeds) / len(saccade_speeds)
                              if saccade_speeds else 0.0),
        "accuracy": (correct_trials / max(1, trial_count)) * 100,
        "trials": trial_count,
    }
    res["score"] = compute_gaze_score(res)
    res["_stalled"] = stalled
    res["_correct_trials"] = correct_trials
    res["_n_timeouts"] = sum(1 for t in trials if t["response"] is None)
    res["_n_incorrect"] = sum(1 for t in trials
                              if t["response"] is not None and not t["correct"])
    return res


MODES = {
    "old":       run(False, False),
    "fix1_only": run(True,  False),
    "fix2_only": run(False, True),
    "both":      run(True,  True),
}


def row(m):
    r = MODES[m]
    s = r["score"]
    return (r["avg_reaction_time"], r["avg_saccade_speed"], r["accuracy"],
            r["trials"], s["score"], s["level"])


print("=== GAZE INPUT ATTRIBUTION (one identical simulated participant) ===\n")
hdr = f"{'mode':<11}{'RT (s)':>9}{'speed':>10}{'acc %':>8}{'trials':>8}{'score':>8}  level"
print(hdr)
print("-" * len(hdr))
for m in ("old", "fix1_only", "fix2_only", "both"):
    rt, sp, ac, tr, sc, lv = row(m)
    print(f"{m:<11}{rt:>9.4f}{sp:>10.4f}{ac:>8.1f}{tr:>8}{sc:>8.1f}  {lv}")

print("\nold-mode trials that never completed (no timeout existed): "
      f"{MODES['old']['_stalled']}")
print(f"both-mode: {MODES['both']['_correct_trials']} correct, "
      f"{MODES['both']['_n_incorrect']} incorrect, "
      f"{MODES['both']['_n_timeouts']} timeouts")

# ---- per-input attribution ------------------------------------------------
o, f1, f2, b = (MODES[k] for k in ("old", "fix1_only", "fix2_only", "both"))
print("\n=== WHICH FIX MOVED WHICH INPUT ===")
for label, key in (("avg_reaction_time", "avg_reaction_time"),
                   ("avg_saccade_speed", "avg_saccade_speed"),
                   ("accuracy", "accuracy"),
                   ("trials", "trials")):
    d1 = abs(f1[key] - o[key]) > 1e-9
    d2 = abs(f2[key] - o[key]) > 1e-9
    who = ("both fixes" if d1 and d2 else
           "trial-advance" if d1 else "iris-reset" if d2 else "neither (unchanged)")
    print(f"  {label:<20} {o[key]:>10.4f} -> {b[key]:<10.4f}  moved by: {who}")

# ---- fusion arithmetic ----------------------------------------------------
BLINK, VOICE = 36.6667, 55.0
print("\n=== FUSION ===")
for m in ("old", "both"):
    g = MODES[m]["score"]["score"]
    ov = compute_overall_score(eye_score=BLINK, gaze_score=g, voice_score=VOICE)
    w = ov["weights"]
    print(f"  {m:<6} gaze={g:<6.1f}  "
          f"{BLINK:.4f}*{w['eye']} + {g:.1f}*{w['gaze']} + {VOICE:.1f}*{w['voice']}"
          f" = {BLINK*w['eye'] + g*w['gaze'] + VOICE*w['voice']:.4f}"
          f"  -> {ov['score']} ({ov['level']})")

# ---- cross-check the simulator against the REAL production method ---------
import tkinter as tk
from tkinter import messagebox
messagebox.showinfo = lambda *a, **k: None
messagebox.showerror = lambda *a, **k: print("   [ERROR DIALOG]", a)
from gui_app import DementiaDetectionGUI
from crat_events import SessionEvents

root = tk.Tk(); root.withdraw()
app = DementiaDetectionGUI(root)
app.gaze_events = SessionEvents()
app.gaze_results = {"reaction_times": [], "saccade_speeds": [], "v_peaks": [],
                    "avg_reaction_time": 0.0, "avg_saccade_speed": 0.0,
                    "accuracy": 0, "trials": 0}
app.trial_count = app.correct_trials = 0
app.total_trials = len(SCRIPT)
app.gaze_running = True
app.gaze_results["saccade_speeds"] = list(b["saccade_speeds"])

# replay the same completions through the real helper
cursor = 0
for i, (target, behaviour, t_first, t_correct) in enumerate(SCRIPT):
    app.current_stimulus = target
    app._trial_speed_start = cursor
    tr = b_trials = None
    resp = None
    if behaviour == "no_response":
        resp, rea = None, None
    elif behaviour == "wrong_then_correct":
        resp, rea = opposite(target), t_first
    else:
        resp, rea = target, t_first
    app._complete_gaze_trial(resp, rea)

real_acc = (app.correct_trials / max(1, app.trial_count)) * 100
print("\n=== CROSS-CHECK vs REAL _complete_gaze_trial() ===")
ok = (app.trial_count == b["trials"] and abs(real_acc - b["accuracy"]) < 1e-9
      and app.gaze_events.summary()["timeouts"] == b["_n_timeouts"])
print(f"  simulator: trials={b['trials']} acc={b['accuracy']:.1f} timeouts={b['_n_timeouts']}")
print(f"  real     : trials={app.trial_count} acc={real_acc:.1f} "
      f"timeouts={app.gaze_events.summary()['timeouts']}")
print("  " + ("PASS — simulator matches production" if ok else "FAIL — simulator diverged"))
root.destroy()

with open(os.path.join(BASE, 'gaze_attribution.json'), 'w', encoding='utf-8') as f:
    json.dump({m: {k: v for k, v in MODES[m].items()
                   if k not in ('saccade_speeds',)} for m in MODES},
              f, indent=2, sort_keys=True, default=str)
    f.write("\n")
print("\nwrote gaze_attribution.json")
sys.exit(0 if ok else 1)
