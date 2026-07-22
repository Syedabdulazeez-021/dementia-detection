# 🧠 Cognitive Risk Assessment (BIO)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.21-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow)

A desktop tool that estimates cognitive-risk indicators from three neurological signals — blink patterns, gaze behaviour, and voice acoustics — using nothing but a webcam and a microphone. Everything runs offline.

| Channel | What it measures | How |
|---|---|---|
| 👁 **Blink** | blink rate, irregularity, eye openness, micro-sleeps, partial blinks | MediaPipe Face Mesh + Eye-Aspect-Ratio |
| 🎯 **Gaze** | reaction time, accuracy, saccade speed | 10-trial stimulus reaction task |
| 🎙 **Voice** | 21 acoustic features → dementia-likeness | Cookie-Theft description + Random Forest |

> ⚠️ **This is a research screening aid, not a medical diagnosis.** A higher score means more behavioural indicators are present — not a confirmed condition. Please consult a clinician for anything clinical.

---

## Table of Contents

* [Highlights](#-highlights)
* [Quick Start](#-quick-start)
* [Workflow](#-workflow)
* [The One Combined Score](#-the-one-combined-score)
* [Explainable AI](#-explainable-ai)
* [Machine Learning & the Feedback Loop](#-machine-learning--the-feedback-loop)
* [Outputs](#-outputs)
* [Project Structure](#-project-structure)
* [Troubleshooting](#-troubleshooting)
* [Limitations](#-limitations)

---

## ✨ Highlights

- Three independent channels (blink, gaze, voice) fused into a single risk score. You can also run just one or two if needed — the score adapts.
- Every feature's contribution to the final score is shown in a bar chart that sums exactly to the overall result. No black box.
- Live plots during the gaze test; a 4-panel voice dashboard (pitch, speech energy + vocal brightness, voice stability, top-8 z-scores) at the end of the voice test.
- The gaze task runs **full-screen**, so the left/right targets sit at a realistic peripheral angle rather than inside a small preview. It closes itself once the 10 trials are done and drops you back to the gaze page.
- One-click PDF report per patient — **every graph from all three tests included**, and the numbers are presented as charts with clinical thresholds drawn on them rather than as bare figures.
- The voice channel is a trained Random Forest; blink and gaze are rule-based by default but will automatically switch to ML models if you drop trained ones in. See the feedback loop section.
- No internet connection needed. Ever.

---

## 🚀 Quick Start

> **Use 64-bit Python 3.10, 3.11, or 3.12.** MediaPipe's `solutions` API isn't available on 3.13 yet.

### Ubuntu / Debian — install system packages first

```bash
sudo apt update
sudo apt install python3-tk portaudio19-dev python3-dev
```

`python3-tk` is for the GUI, `portaudio19-dev` for mic input, and `python3-dev` is needed by some audio packages at build time.

### Windows

```bash
py -3.11 -m venv venv
venv\Scripts\activate
python --version
pip install -r requirements.txt
python main.py
```

### Ubuntu / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python --version
pip install -r requirements.txt
python main.py
```

`requirements.txt` pins `mediapipe==0.10.21`. The three model files (`models/eye_classifier_best.h5`, `dementia_rf_model.pkl`, `scaler.pkl`) are bundled and need to stay where they are.

---

## 🩺 Workflow

1. Register the patient.
2. **Blink test (60 s)** — patient faces the camera; the first ~5 s calibrate a baseline automatically.
3. **Gaze test (10 trials)** — opens full-screen for a quick 3-point calibration, then the patient looks at each target as it appears. The window closes on its own after the 10th trial and returns to the gaze page, where the reaction-time and saccade-speed graphs are waiting. Press `Esc` to stop early.
4. **Voice test (55 s)** — patient describes the Cookie-Theft picture out loud; dashboard is generated at the end. **Cancel Analysis** aborts the recording immediately if needed.
5. **Results** — per-test scores, the fused overall risk, and the "Why this score?" explainability panel.
6. **Export** — CSV entry written automatically; PDF report on demand.

---

## 🧮 The One Combined Score

```
overall = Σ (testᵢ score × weightᵢ) / Σ weightᵢ

weights:
  voice = 0.40
  blink = 0.35
  gaze  = 0.25
```

If a test is skipped the remaining weights renormalise, so the score is still meaningful with a partial session.

These weights come from the literature, not from training data. If you have a labelled same-subject cohort, `ml/train_fusion.py` can learn them instead.

---

## 🔍 Explainable AI

The results page and PDF both show a bar chart where each bar is one feature's contribution to the final score, colour-coded by channel. Because the fusion is linear, those bars sum exactly to the overall number.

The Random Forest voice score is split across its top driving features using an importance-based attribution. It's approximate (not SHAP), and it's labelled that way.

---

## 🤖 Machine Learning & the Feedback Loop

Voice is the only channel currently using a trained model. Blink and gaze are rule-based for now, but if you place a trained model file in the right location they'll switch to it automatically — the results page will show an `[ML]` tag so you know which one is running.

Every session logs its raw features to three CSV files:

- `retrain_data.csv`
- `blink_retrain_data.csv`
- `gaze_retrain_data.csv`

Each has a blank `true_label` column. Once a diagnosis is confirmed, fill it in:

```
0 = normal
1 = dementia
```

Then retrain:

```bash
python ml/train_models.py --modality blink --data blink_retrain_data.csv
python ml/train_models.py --modality gaze  --data gaze_retrain_data.csv
python ml/train_fusion.py --data multimodal_labelled.csv
python retrain.py
```

The model only ever learns from confirmed ground truth — not from its own previous predictions. That's intentional. See `ml/README.md` for CSV formats and pointers to real labelled datasets.

---

## 📁 Outputs

| File | What's in it |
|---|---|
| `patients_record.csv` | Central record — all scores + top risk driver per session |
| `report_<token>_<timestamp>.pdf` | Per-patient clinical report — see below |
| `*_retrain_data.csv` | Feature logs for building a labelled dataset over time |
| `waveform_plot.png` | Raw audio waveform from the voice test |
| `feature_plot.png` | Feature bar chart |
| `voice_dashboard.png` | The 4-panel voice dashboard |

### What's in the PDF report

Generated by **📄 GENERATE PDF REPORT** on the results page. Pages appear only for the tests that were actually run, so a partial session produces a shorter report.

| Page | Contents |
|---|---|
| 1 — Summary | Patient details, overall risk banner, per-test score table, **Top drivers** bar chart (with a "major driver ≥ 10 pts" threshold line), and a **measured-vs-normal-range** chart |
| 2 — Blink & gaze graphs | Eye openness over time, blink regularity, gaze reaction time per trial, saccade speed per trial |
| 3+ — Voice graphs | The voice dashboard, waveform/speech segmentation, and feature comparison charts |

The two page-1 charts are the explainability core. **Top drivers** shows each feature's contribution in points, colour-coded by channel. **Measured vs normal range** puts blink rate, eye openness, gaze reaction time, gaze accuracy, saccade speed and voice risk each on its own track, with the healthy band shaded, the threshold marked, and the patient's value plotted as a diamond — teal inside the range, red outside. Thresholds are read from `scoring.py` and `dementia_analyzer.py`, so the chart and the score can't drift apart.

---

## 🗂 Project Structure

```
main.py                       Entry point → launches the GUI
gui_app.py                    Tkinter GUI: pages, live graphs, full-screen gaze task, PDF report
gui_adapter.py                Bridges webcam/detector with the analyzer
mediapipe_detector.py         MediaPipe 468-landmark eye detector (EAR)
dementia_analyzer.py          Blink/eye risk scoring + calibration
gaze_stimulus_experiment.py   Gaze reaction-time logic
voice_dimentia.py             Voice pipeline (21 features → Random Forest) + dashboard
scoring.py                    Gaze score, fusion, explainability, optional ML model loading
retrain.py                    Voice feedback-retraining (confirmed labels)
ml/                           Training pipeline (train_models.py, train_fusion.py, templates)
docs/                         HOW_IT_WORKS.md, REFERENCES.md
models/eye_classifier_best.h5  Eye-state CNN
dementia_rf_model.pkl         Voice Random Forest   (repo root)
scaler.pkl                    Voice feature scaler  (repo root)
```

`docs/HOW_IT_WORKS.md` walks through the full system and explains every graph. Scientific references are in `docs/REFERENCES.md`.

---

## 🔧 Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**

You're on Python 3.13+. MediaPipe removed the legacy `mp.solutions` API in 0.10.31, and the versions that still have it (including the pinned `0.10.21`) ship no 3.13 wheel — so pip silently installs a newer, incompatible one. Use Python 3.10–3.12:

```bash
py -3.11 -m venv venv          # Windows
python3.11 -m venv venv        # Linux/macOS
```

Check what you're actually running with `py -0p` (Windows) or `python3 --version`. Note that activating the venv is what makes `python` resolve to the right interpreter — outside it, plain `python` may still be 3.13.

**The GUI takes 10–20 s to appear** — that's TensorFlow importing. Normal on first launch.

**Camera or microphone not found** — close anything else using the device (Zoom, Teams, browser tabs) and allow access when the OS prompts. On Linux, make sure your user is in the `video` and `audio` groups.

**The voice test shows a placeholder instead of a picture** — `cookie_theft.jpeg` is missing from the repo root. Any Cookie-Theft image named `cookie_theft.jpg`, `.png`, or `.jpeg` will be picked up.

---

## ⚖ Limitations

Worth being upfront about:

- Fusion weights and blink/gaze thresholds are heuristic, not learned from a clinical dataset.
- Webcam-based gaze is a practical proxy for proper saccade testing — convenient, but not the same thing.
- The voice model was trained on a small, single-accent dataset. Specificity is decent; sensitivity less so.
- Performance degrades with poor lighting, thick-framed glasses, or a low-quality microphone.
- The combined system hasn't been validated on a labelled clinical cohort yet.

Use this as a screening or research tool, not a diagnostic one.
## Voice test stimulus

The voice test uses the Cookie Theft picture from the Boston Diagnostic
Aphasia Examination (Goodglass & Kaplan, 1983). The image is copyrighted
and is not distributed with this repository.

Place your own licensed copy at the repository root as `cookie_theft.jpeg`
before running the voice test.
