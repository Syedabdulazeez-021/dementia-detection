"""dev/verify_gaze_fixes.py - behavioural checks for the gaze trial loop.

Exercises `DementiaDetectionGUI._complete_gaze_trial()`, the real production
method, with a scripted mix of correct, incorrect and timed-out trials.

Asserts that a trial advances on any classified response (not only a correct
one), that the per-trial timeout completes a trial as incorrect with no
reaction time, that one V_peak is recorded per trial, and that the iris
position is reset at stimulus onset. No camera needed.

    python dev/verify_gaze_fixes.py
"""

import os
import sys

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import tkinter as tk
from tkinter import messagebox

messagebox.showinfo = lambda *a, **k: None
messagebox.showwarning = lambda *a, **k: None
messagebox.showerror = lambda *a, **k: print("   [ERROR DIALOG]", a)

import gui_app
from gui_app import DementiaDetectionGUI, GAZE_TRIAL_TIMEOUT_S
from crat_events import SessionEvents

FAILS = []


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(label)


root = tk.Tk()
root.withdraw()
app = DementiaDetectionGUI(root)

# ---------------------------------------------------------------------------
# Scripted 10-trial session: 6 correct, 2 deliberately-wrong, 2 timeouts.
# ---------------------------------------------------------------------------
SCRIPT = [
    ("LEFT",  "LEFT",  0.42),
    ("RIGHT", "RIGHT", 0.51),
    ("LEFT",  "RIGHT", 0.83),   # wrong direction
    ("RIGHT", "RIGHT", 0.47),
    ("LEFT",  None,    None),   # timeout
    ("RIGHT", "LEFT",  0.91),   # wrong direction
    ("LEFT",  "LEFT",  0.55),
    ("RIGHT", "RIGHT", 0.49),
    ("LEFT",  None,    None),   # timeout
    ("RIGHT", "RIGHT", 0.58),
]

app.gaze_events = SessionEvents()
app.gaze_results = {"avg_reaction_time": 0.0, "avg_saccade_speed": 0.0,
                    "accuracy": 0, "trials": 0, "reaction_times": [],
                    "saccade_speeds": [], "v_peaks": []}
app.trial_count = 0
app.correct_trials = 0
app.total_trials = len(SCRIPT)
app.gaze_running = True

for i, (target, response, reaction) in enumerate(SCRIPT):
    app.current_stimulus = target
    app._trial_speed_start = len(app.gaze_results["saccade_speeds"])
    if response is not None:
        # a few per-frame speed samples for this trial
        app.gaze_results["saccade_speeds"].extend([12.0 + i, 30.0 + i, 8.0 + i])
    # a timed-out trial may have no samples at all
    app._complete_gaze_trial(response, reaction)

ev = app.gaze_events
s = ev.summary()

print("\n=== GAZE TRIAL-LOOP BEHAVIOURAL CHECKS ===")
print(f"  events summary: {s}")

check("trial advance: every trial advances trial_count (not only correct ones)",
      app.trial_count == 10, f"trial_count={app.trial_count}")
check("trial advance: correct_trials counts only matches",
      app.correct_trials == 6, f"correct_trials={app.correct_trials}")
check("trial advance: accuracy is no longer pinned at 100 %",
      abs(app.correct_trials / app.trial_count * 100 - 60.0) < 1e-9,
      f"{app.correct_trials / app.trial_count * 100:.1f} %")
check("timeout: timeout constant exists and defaults to 3.0 s",
      GAZE_TRIAL_TIMEOUT_S == 3.0, f"GAZE_TRIAL_TIMEOUT_S={GAZE_TRIAL_TIMEOUT_S}")
check("timeout: timed-out trials contribute NO reaction time",
      len(app.gaze_results["reaction_times"]) == 8,
      f"{len(app.gaze_results['reaction_times'])} RTs for 10 trials, 2 timeouts")
check("events: incorrect trials are logged as events",
      s["trials"] == 10 and s["correct"] == 6,
      f"trials={s['trials']} correct={s['correct']}")
check("events: timeouts are logged and flagged",
      s["timeouts"] == 2, f"timeouts={s['timeouts']}")
check("events: events accuracy matches the counters",
      abs(s["accuracy"] - app.correct_trials / app.trial_count * 100) < 1e-9,
      f"events={s['accuracy']:.1f} % counters={app.correct_trials / app.trial_count * 100:.1f} %")
check("V_peak: one V_peak per trial, incl. timeouts",
      len(ev.v_peaks()) == 10, f"{len(ev.v_peaks())} V_peak values")
check("V_peak: a timed-out trial with no samples yields V_peak 0.0",
      ev.v_peaks()[4] == 0.0 and ev.v_peaks()[8] == 0.0,
      f"T5={ev.v_peaks()[4]} T9={ev.v_peaks()[8]}")

timeout_trials = [t for t in ev.trials if t.get("timeout")]
check("V_peak: timed-out trials carry response=None and correct=False",
      all(t["response"] is None and t["correct"] is False for t in timeout_trials),
      f"{len(timeout_trials)} timeout records")
wrong = [t for t in ev.trials if not t["correct"] and not t.get("timeout")]
check("events: incorrect trials keep a real response side",
      len(wrong) == 2 and all(t["response"] in (-1, 1) for t in wrong),
      f"{len(wrong)} incorrect records")

# ---------------------------------------------------------------------------
# Iris reset: assert the source clears prev_eye_x at stimulus onset.
# ---------------------------------------------------------------------------
src = open(os.path.join(BASE, 'gui_app.py'), encoding='utf-8').read()
onset = src.index('self.stimulus_time = time.time()')
window = src[onset:onset + 1200]
check("reset: prev_eye_x/prev_time reset at stimulus onset",
      'self.prev_eye_x = None' in window and 'self.prev_time = None' in window)
check("reset: V(t) formulas untouched",
      'dist = abs(eye_x - self.prev_eye_x)' in src and 'speed = dist / dt' in src)
check("reset: int() cast on mesh_points untouched",
      '[(int(p.x*w), int(p.y*h)) for p in results.multi_face_landmarks[0].landmark]' in src)

# ---------------------------------------------------------------------------
# Panel (c) must render the new states.
# ---------------------------------------------------------------------------
import crat_figures as CF
out = os.path.join(BASE, '_chk_gaze_fixes.png')
try:
    CF.gaze_figure(ev, gaze=app.gaze_results, out_path=out, dpi=72)
    check("figure: panel (c) renders a mixed correct/incorrect/timeout session",
          os.path.exists(out))
    os.remove(out)
except Exception as e:
    check("figure: panel (c) renders a mixed correct/incorrect/timeout session", False, str(e))

print("\n=== RESULT ===")
print("all checks passed" if not FAILS else "FAILED:\n  " + "\n  ".join(FAILS))
root.destroy()
sys.exit(1 if FAILS else 0)
