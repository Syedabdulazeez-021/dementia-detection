# 🧠 Cognitive Risk Assessment (BIO)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.21-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow)

A non-invasive, fully-offline desktop tool that estimates **cognitive-risk
indicators** from three independent neurological signals captured with a normal
webcam and microphone, fuses them into **one overall score**, and explains
exactly how that score was reached.

| Channel | What it measures | How |
|---------|------------------|-----|
| 👁 **Blink** | blink rate, irregularity, eye openness, micro-sleeps, partial blinks | MediaPipe Face Mesh + Eye-Aspect-Ratio |
| 🎯 **Gaze** | reaction time, accuracy, saccade speed | 10-trial stimulus reaction task |
| 🎙 **Voice** | 21 acoustic features → dementia-likeness | Cookie-Theft description + Random Forest |

> ⚠ **Disclaimer** — This is a *research screening aid, not a medical diagnosis.*
> A higher score means more behavioural indicators are present, not a confirmed
> condition. Always consult a qualified clinician.

---

## Table of Contents
- [Highlights](#-highlights)
- [Quick Start](#-quick-start)
- [Workflow](#-workflow)
- [The One Combined Score](#-the-one-combined-score)
- [Explainable AI](#-explainable-ai)
- [Machine Learning & the Feedback Loop](#-machine-learning--the-feedback-loop)
- [Outputs](#-outputs)
- [Project Structure](#-project-structure)
- [Limitations](#-limitations)

---

## ✨ Highlights

- **Multimodal** — blink + gaze + voice in a single session; any subset can be run.
- **One fused score** — a weighted overall risk that adapts when a test is skipped.
- **Explainable** — a per-feature contribution chart whose bars **sum exactly to the final score**, including an approximate attribution of the Random-Forest voice decision.
- **Live graphs** — reaction-time and saccade-speed plots build up during the gaze test; a 4-panel voice dashboard (pitch, merged speech-energy + vocal-brightness, voice stability, top-8 z-scores).
- **PDF report** — one click produces a per-patient clinical report.
- **ML-ready for every channel** — a training pipeline + a feedback loop that learns from clinician-confirmed labels (see below).
- **Fully offline** — no network needed; appropriate for sensitive data.

---

## 🚀 Quick Start

> **Use 64-bit Python 3.10, 3.11, or 3.12 — NOT 3.13.**
> (MediaPipe's legacy `solutions` API used here is unavailable on 3.13.)

```bash
py -3.11 -m venv venv          # or -3.12 / -3.10
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
python --version               # confirm 3.10–3.12
pip install -r requirements.txt
python main.py
```

`requirements.txt` pins `mediapipe==0.10.21`. The three model files
(`models/eye_classifier_best.h5`, `dementia_rf_model.pkl`, `scaler.pkl`) are
included and required.

---

## 🩺 Workflow

1. **Register** the patient.
2. **Blink test (60 s)** — sit facing the camera; first ~5 s auto-calibrate your baseline.
3. **Gaze test (10 trials)** — after a 3-point calibration, look at the target that appears; live graphs update each trial.
4. **Voice test (55 s)** — describe the Cookie-Theft picture; a voice dashboard is produced.
5. **Results** — per-test scores, the single overall risk, the "Why this score?" explainability panel, and graphs.
6. **Export** — CSV record and a one-click PDF report.

---

## 🧮 The One Combined Score

The overall risk is a weighted average of the tests performed:

```
overall = Σ (testᵢ score × weightᵢ) / Σ weightᵢ      weights: voice 0.40, blink 0.35, gaze 0.25
```

Weights renormalise when a test is skipped, so the single score is always valid.
These weights are **literature-informed defaults, not learned from data** — see
`ml/train_fusion.py` to learn them from a labelled same-subject cohort.

---

## 🔍 Explainable AI

The results page and PDF report show a bar chart of **each feature's contribution
to the final score**, colour-coded by channel, with plain-language reasons.
Because the fusion is linear, the bars sum exactly to the overall score. The
Random-Forest voice score is split across its top driving features using an
**approximate** importance-based attribution (labelled as such, not exact SHAP).

---

## 🤖 Machine Learning & the Feedback Loop

- **Voice** is a trained Random Forest. **Blink** and **gaze** are rule-based by
  default but **ML-ready**: if a trained model file is present they are used
  automatically (shown with an `[ML]` tag); if absent, the rule-based score is used.
- Every session logs its features to `retrain_data.csv` (voice),
  `blink_retrain_data.csv`, and `gaze_retrain_data.csv`, each with a blank
  `true_label` column. When a real diagnosis is later **confirmed**, fill in
  `true_label` (0 = normal, 1 = dementia).
- Train real models from that labelled data:

  ```bash
  python ml/train_models.py --modality blink --data blink_retrain_data.csv
  python ml/train_models.py --modality gaze  --data gaze_retrain_data.csv
  python ml/train_fusion.py  --data multimodal_labelled.csv   # learns the single score
  python retrain.py                                            # retrains voice from confirmed labels
  ```

  See `ml/README.md` for CSV formats and how to obtain real labelled datasets.

> The model learns only from **confirmed ground truth**, never from its own
> predictions. It improves once enough labelled cases accumulate — this is the
> correct, honest design, not a self-improving model.

---

## 📁 Outputs

- `patients_record.csv` — central record (all scores + top risk driver).
- `report_<token>_<timestamp>.pdf` — per-patient report.
- `*_retrain_data.csv` — feature logs for building a labelled dataset.
- `waveform_plot.png`, `feature_plot.png`, `voice_dashboard.png` — voice visuals.

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
models/eye_classifier_best.h5,  dementia_rf_model.pkl,  scaler.pkl
```

A detailed walkthrough of the system and the meaning of every graph is in
[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md). Scientific references are in
[`docs/REFERENCES.md`](docs/REFERENCES.md).

---

## ⚖ Limitations

Heuristic thresholds and fusion weights (not learned); webcam gaze is an
accessible proxy for clinical saccade tests; the voice model is trained on a
small, single-accent dataset (good specificity, moderate sensitivity); sensitive
to lighting, eyeglasses, and microphone quality; the combined system is **not
yet validated on a labelled clinical cohort**. Use as a screening / research
demonstrator only.
