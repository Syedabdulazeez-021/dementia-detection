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
* [Limitations](#-limitations)

---

## ✨ Highlights

- Three independent channels (blink, gaze, voice) fused into a single risk score. You can also run just one or two if needed — the score adapts.
- Every feature's contribution to the final score is shown in a bar chart that sums exactly to the overall result. No black box.
- Live plots during the gaze test; a 4-panel voice dashboard (pitch, speech energy + vocal brightness, voice stability, top-8 z-scores) at the end of the voice test.
- One-click PDF report per patient.
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
3. **Gaze test (10 trials)** — quick 3-point calibration, then look at each target as it appears. Reaction time and saccade speed graphs build up live.
4. **Voice test (55 s)** — patient describes the Cookie-Theft picture out loud; dashboard is generated at the end.
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
| `report_<token>_<timestamp>.pdf` | Per-patient clinical report |
| `*_retrain_data.csv` | Feature logs for building a labelled dataset over time |
| `waveform_plot.png` | Raw audio waveform from the voice test |
| `feature_plot.png` | Feature bar chart |
| `voice_dashboard.png` | The 4-panel voice dashboard |

---

## 🗂 Project Structure

```
main.py                       Entry point → launches the GUI
gui_app.py                    Tkinter GUI: pages, live graphs, results, PDF report
gui_adapter.py                Bridges webcam/detector with the analyzer
mediapipe_detector.py         MediaPipe 468-landmark eye detector (EAR)
dementia_analyzer.py          Blink/eye risk scoring + calibration
gaze_stimulus_experiment.py   Gaze reaction-time logic
voice_dimentia.py             Voice pipeline (21 features → Random Forest) + dashboard
scoring.py                    Gaze score, fusion, explainability, optional ML model loading
retrain.py                    Voice feedback-retraining (confirmed labels)
ml/                           Training pipeline (train_models.py, train_fusion.py, templates)
docs/                         HOW_IT_WORKS.md, REFERENCES.md
models/eye_classifier_best.h5
models/dementia_rf_model.pkl
models/scaler.pkl
```

`docs/HOW_IT_WORKS.md` walks through the full system and explains every graph. Scientific references are in `docs/REFERENCES.md`.

---

## ⚖ Limitations

Worth being upfront about:

- Fusion weights and blink/gaze thresholds are heuristic, not learned from a clinical dataset.
- Webcam-based gaze is a practical proxy for proper saccade testing — convenient, but not the same thing.
- The voice model was trained on a small, single-accent dataset. Specificity is decent; sensitivity less so.
- Performance degrades with poor lighting, thick-framed glasses, or a low-quality microphone.
- The combined system hasn't been validated on a labelled clinical cohort yet.

Use this as a screening or research tool, not a diagnostic one.