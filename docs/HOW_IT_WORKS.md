# Cognitive Risk Assessment — How It Works (Full Explanation)

This document explains the whole system end to end, then the meaning of **every
graph** the app produces, in plain language.

> Reminder: this is a **screening aid, not a diagnosis**. Every score reflects
> how many behavioural *indicators* are present — not a confirmed condition.

---

## 1. The big picture

A single session captures three independent signals from a normal webcam and
microphone, and turns each into a risk score from **0 to 100** (0 = no
indicators, 100 = strong indicators):

| Test | What it captures | How long |
|------|------------------|----------|
| **Blink** | how you blink (rate, regularity, eye openness, micro-sleeps) | 60 s |
| **Gaze** | how fast/accurately your eyes react to a target | 10 trials |
| **Voice** | acoustic features of your speech | 55 s recording |

The three scores are then **fused** into one overall risk, and an
**explainability** layer shows exactly how much each feature contributed.

Pipeline: `camera/mic → features → 3 scores → weighted fusion → overall risk + explanation → report`

---

## 2. How each test is designed (protocol)

**Blink test (60 seconds).** You sit facing the camera in reasonable light.
MediaPipe Face Mesh tracks 468 face landmarks; six eyelid points per eye give
the **Eye-Aspect-Ratio (EAR)** — a single number that drops when the eye
closes. The first ~5 seconds calibrate *your* normal open-eye value, and the
blink threshold is set to 85% of that baseline (so it adapts to each person).
Over the minute it counts blinks, measures how regular the gaps are, how open
the eyes stay, long closures (>0.5 s = "micro-sleeps"), and incomplete
("partial") blinks.

**Gaze test (10 trials).** After a short left/centre/right calibration, each
trial shows a fixation point, then a target appears on one side. The system
measures: **reaction time** (how long until your eyes move to it), **accuracy**
(did you look the correct way), and **saccade speed** (how fast the eye jumped).
This is an accessible webcam proxy for the clinical "saccade" tests used in
dementia research.

**Voice test (55 seconds, Cookie-Theft-style description).** You describe a
picture out loud for 55 seconds (recorded mono at 22,050 Hz). The audio is
reduced to **21 acoustic features** in four groups — spectral shape (MFCC
statistics), Mel filter-bank energies, pause/silence behaviour, and
voice-quality (jitter, shimmer, harmonics-to-noise ratio). A trained **Random
Forest** then outputs the probability of dementia-like speech.

---

## 3. The meaning of every graph

### 3a. Gaze experiment page (live)

- **Reaction Time per Trial** (top): one orange dot per trial = how many
  seconds your eyes took to reach the target. The **green shaded band (≤0.5 s)**
  is the healthy zone; the **red dotted line** is your running average. Dots
  rising above the band = slowing responses.
- **Saccade Speed** (bottom): the speed of eye movement across samples. The
  **green dashed line at 300 px/s** marks the healthy floor; the red dotted line
  is the average. Consistently low values = sluggish eye movement.

### 3b. Results page

- **Live EAR / blink graphs**: eye-openness over time with the adaptive
  threshold; dips below the line are blinks, long dips are micro-sleeps.
- **Blink summary graph**: blink rate vs the normal 12–25 bpm band.
- **Gaze graphs**: same reaction-time and saccade-speed views as above.
- **"Why this score?" (Explainable AI) chart**: a horizontal bar per feature
  showing **how many points it added to the final score**, colour-coded
  (blue = blink, orange = gaze, purple = voice). The bars **add up exactly to
  the overall score** — nothing hidden. The text beside it names the top drivers
  in plain words.

### 3c. Voice dashboard (now 4 panels)

- **Pitch (F0) Variation**: your vocal pitch over time (blue), with the mean
  (gold dashed). Red-shaded spots mark unusually low-pitch moments. Very flat or
  very erratic pitch can both be informative.
- **Speech Energy & Vocal Brightness (merged)**: this is the panel you asked to
  combine. **Green = speech energy (RMS)** on the left axis — tall green =
  talking, flat green = quiet. **Orange = vocal brightness (spectral centroid)**
  on the right axis — higher = clearer/sharper voice. The **red-shaded vertical
  bands are silent gaps** (no energy spike). Reading them together: long/frequent
  silences plus low brightness during speech are the patterns of interest. The
  title reports speech rate (syllables/s) and total % silence.
- **Voice Stability (jitter / shimmer / HNR)**: your values (orange) next to the
  training **AD average (red)** and **CN average (green)**. If your bars sit near
  the red bars, your voice resembles the dementia group; near green = healthy.
- **Top-8 Discriminant Features (Z-scores)**: how far each of the most important
  features is from the training norm, in standard deviations. Bars beyond **±1.5σ
  (red dotted lines)** are flagged abnormal (red); near zero = normal (green).

### 3d. The workflow diagram (paper Fig. 1)

A schematic of the pipeline above: one session → three branches → three scores →
weighted fusion → explainability → report.

---

## 4. How the scores are combined (and the weights)

Inside each test, every feature has a maximum point value (its "importance"),
and how abnormal your measurement is decides how much it earns. The three test
scores are then combined with fixed weights:

- **Voice = 0.40, Blink = 0.35, Gaze = 0.25** (they sum to 1).
- Voice is weighted highest because it is a *trained model*; blink and gaze are
  rule-based.
- If a test is skipped, its weight is dropped and the rest are rescaled, so the
  overall score stays valid.

`overall = (Σ weightₘ × scoreₘ) / (Σ weightₘ)` over the tests performed.

**Honest note:** these weights and point values are **chosen, literature-informed
defaults — not learned from data**. That is fine for a prototype, and learning
them from a labelled cohort is the planned next step.

---

## 5. Explainability — why the numbers are trustworthy

Because the fusion is linear, each feature's contribution to the final score is
just `its points × its test's weight`, and all contributions sum exactly to the
overall score. For the Random-Forest voice score (which isn't additive by
nature), the contribution is split across its top features using the model's
feature importance × how AD-like each feature is — labelled "approximate"
because it is an estimate, not exact SHAP.

---

## 6. "Learning from feedback" (the ML loop)

A model can only truly learn from **confirmed** answers, never from its own
guesses. So:

1. Every voice session is logged (21 features + prediction) to
   `retrain_data.csv` with a blank `true_label` column.
2. When a real diagnosis is later confirmed, you fill in `true_label`
   (0 = healthy, 1 = dementia) for that row.
3. Running `python retrain.py` trains a fresh Random Forest on those confirmed
   rows, cross-validates it, compares it to the current model, backs up the old
   one, and **only replaces it if it scores higher**.

It won't improve anything until enough confirmed labels accumulate — that is
expected and honest, not a flaw.

---

## 7. Limitations (state these plainly)

Heuristic thresholds and weights (not learned); webcam gaze is a proxy for lab
saccade tests; the voice model is trained on a small, single-accent dataset
(good specificity, moderate sensitivity); sensitive to lighting, eyeglasses, and
microphone quality; the combined system is **not yet validated on a labelled
clinical cohort**. Use as a screening/research demonstrator only.
