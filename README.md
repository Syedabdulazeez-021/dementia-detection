# Cognitive Risk Assessment (CRAT)

## 1. What this is

CRAT is a desktop application that looks for early signs of cognitive decline
using only a normal laptop webcam and microphone. It runs three short tests,
scores each one from 0 to 100, and combines them into a single risk number with
an explanation of what produced it.

Everything runs on the computer itself. No internet connection is used and no
data leaves the machine.

**This is a research prototype and a screening aid. It is not a medical device.
It does not diagnose anything.** A score from this tool means some measurable
indicators were present. It does not mean a person has dementia. Only a
qualified clinician can make that judgement.

## 2. The three tests

**Blink, 60 seconds.** The person sits still and blinks naturally while the
webcam watches their eyes. The first few seconds are used to calibrate a
baseline. The app measures how often they blink, how regular the gaps between
blinks are, how wide the eyes stay open on average, how many long closures
lasting more than half a second occur, and how many blinks are incomplete. An
incomplete blink is one where the eyelid starts to close but does not fully
shut.

**Gaze, 10 trials.** A dot appears on the left or the right of the screen. The
person looks at it. For each trial the app measures how quickly they reacted,
whether they looked the correct way, and how fast the eye moved. If there is no
response within 3 seconds, the trial ends and counts as incorrect.

**Voice, 55 seconds.** The person describes a picture out loud. The app pulls 21
measurements out of the recording. These cover tone, loudness across different
frequencies, the length and number of pauses, and how steady the voice is.

## 3. How the final score is worked out

Each test produces a score from 0 to 100, where a higher number means more
indicators were found.

The three scores are combined into one using a weighted average. Voice counts
for 0.40, blink for 0.35, and gaze for 0.25. Voice carries the most weight
because it is the only test that uses a trained model. The other two use rules
taken from published research.

```
overall = 0.40 x voice + 0.35 x blink + 0.25 x gaze
```

If a test is skipped, its weight is dropped and the weights of the remaining
tests are rescaled so they still add up to 1. The result stays on the same 0 to
100 scale either way.

The final number falls into one of five bands:

| Score | Band |
|---|---|
| 0 to 20 | LOW |
| 20 to 40 | MILD |
| 40 to 60 | MODERATE |
| 60 to 80 | HIGH |
| 80 to 100 | VERY HIGH |

## 4. Why you can see how it decided

The combination is a plain weighted sum, so it can be taken apart again. The app
reports how many points each individual measurement contributed to the final
number, and those contributions add back up to the final number exactly. There
is no hidden step.

The voice model is the one exception. It is a Random Forest classifier, so its
share of the score cannot be split across its features exactly. The app
approximates the split using the model's feature importances, and labels that
part of the report as approximate.

## 5. What has and has not been tested

Read this before drawing any conclusion from a score.

The **voice classifier** was tested on a held-out split of cognitively normal
versus Alzheimer's recordings. It reached **63.7 percent accuracy, 75 percent
specificity, and 52.5 percent sensitivity**. Sensitivity of 52.5 percent means
it misses close to half of the people it should flag.

The **blink and gaze scores are rule-based**. Their thresholds come from
published ranges. They have no accuracy figure of their own, because they were
never tested against labelled data.

The **combined three-test score has never been tested against a labelled group
of patients.** The weights of 0.40, 0.35 and 0.25 are informed defaults chosen
by hand. They are not learned values, and there is no evidence that this
particular combination performs better than any other.

## 6. Installing and running

Full steps are in [SETUP.md](SETUP.md). The short version:

Python 3.12 is required, because the face-tracking library needs it. Newer
versions of Python will not work.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

The second line is the Windows activation command. If PowerShell blocks it, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then
activate again.

## 7. What you need

A laptop with a webcam and a microphone. Ordinary indoor lighting. Sit about 50
to 60 cm from the screen. No special hardware is required.

## 8. What comes out

A PDF report containing:

- the overall score and its band
- the score for each of the three tests
- the charts from each test, with each test on its own page
- a bar chart showing which individual measurements pushed the score up
- a chart showing each measurement against its normal range

Pages appear only for the tests that were actually run, so a partial session
produces a shorter report. A row is also appended to a local CSV record.

## 9. Project layout

| Path | What it does |
|---|---|
| `main.py` | Entry point. Checks for the model files, then starts the interface. |
| `gui_app.py` | The desktop interface. Runs all three tests, draws the live charts, and builds the PDF report. |
| `dementia_analyzer.py` | Blink detection and the blink score. Finds blinks, long closures, and incomplete blinks from eye measurements. |
| `mediapipe_detector.py` | Turns a webcam frame into eye measurements using a 468-point face mesh. |
| `gui_adapter.py` | Connects the interface to the blink detector and the face mesh. |
| `voice_dimentia.py` | Voice recording, the 21 acoustic measurements, the trained classifier, and the voice charts. |
| `scoring.py` | The gaze score, the weighted combination, the risk bands, and the contribution breakdown. |
| `crat_figures.py` | Shared colours and typography, plus the blink and gaze result figures. |
| `crat_events.py` | Records when each event happened during a session. Held in memory only and never written to disk. |
| `gaze_stimulus_experiment.py` | A standalone version of the gaze task. Not used by the desktop app. |
| `app.py` | A separate web version built on Flask. It measures eye openness a different way, so its scores are not comparable. |
| `ml/` | Scripts and CSV templates for training optional blink, gaze, and fusion models from labelled data. |
| `retrain.py` | Retrains the voice model once confirmed diagnoses have been filled in. |
| `dev/verify_scores.py` | Regression harness. Runs the scoring path with fixed inputs, so any change to a score shows up as a difference. |
| `models/`, `dementia_rf_model.pkl`, `scaler.pkl` | Trained model files loaded at run time. |
| `docs/` | Longer notes on how the system works, and the references behind the thresholds. |

## 10. Known limitations

- The blink rate is worked out by dividing the blink count by wall-clock time
  since the session started, not by the 60 second recording window. If the app
  is left open after the test finishes, the reported rate falls below the true
  value.
- Saccade speed is measured as iris movement in camera pixels, not as a gaze
  angle. The 300 px/s reference it is compared against is therefore not
  meaningful, and head movement is hard to separate from eye movement.
- The gaze reaction times recorded so far are faster than a real eye movement
  can be. The direction detection needs checking before those figures are
  trusted.
- Results depend on lighting, on whether the person wears glasses, and on webcam
  quality.
- The voice model was trained on a small, single-accent dataset. It may perform
  worse on speakers whose accent is not represented in that data.
- Eye closures shorter than half a second are counted, but their durations are
  discarded, so there is no full picture of closure length.
- Blink and gaze thresholds are heuristic. They were taken from published ranges
  rather than learned from a clinical dataset.
- The web version in `app.py` measures eye openness by a different method than
  the desktop app. Scores from the two cannot be compared.
- The system has not been validated on a clinical group. Treat it as a
  demonstrator.

## 11. Voice test picture

The voice test uses the Cookie Theft picture from the Boston Diagnostic Aphasia
Examination (Goodglass and Kaplan, 1983). The app looks for a file named
`cookie_theft.jpeg`, `cookie_theft.jpg`, or `cookie_theft.png` in the repository
root. If none is found, a placeholder is shown instead.

## 12. Licence

See [LICENSE](LICENSE).
