# SETUP.md — running this project

Copy-pasteable setup for Windows. Follow it top to bottom.

---

## Why the Python version matters

This project uses MediaPipe's **legacy `solutions` API** — specifically
`mp.solutions.face_mesh`, at `mediapipe_detector.py:16` and `gui_app.py:706`.

MediaPipe **removed** that API in 0.10.31. Releases that still ship it
(≤ 0.10.21) publish wheels only up to **Python 3.12**. So:

| Python | mediapipe with `solutions` | Works? |
|---|---|---|
| 3.13 / 3.14 | no wheel available | **no** — `AttributeError: module 'mediapipe' has no attribute 'solutions'` |
| **3.12** | **0.10.21** | **yes** |
| 3.10 / 3.11 | 0.10.21 | yes |

If you see this, you are on the wrong Python:

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

**Do not "fix" it by porting the detector to the `tasks` API.** Use Python 3.12.

---

## 1. Confirm Python 3.12 is installed

```powershell
py -0p
```

You should see a `-V:3.12` line. If not, install 64-bit Python 3.12 from
python.org or the Microsoft Store, then re-run the command.

## 2. Create the virtual environment

From the repository root:

```powershell
py -3.12 -m venv .venv312
```

## 3. Activate it (Windows PowerShell)

```powershell
.\.venv312\Scripts\Activate.ps1
```

If PowerShell blocks the script with an execution-policy error, allow it for the
current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
```

Other shells:

```powershell
.\.venv312\Scripts\activate.bat      # cmd.exe
source .venv312/Scripts/activate     # Git Bash
```

Your prompt should now be prefixed `(.venv312)`.

## 4. Install the dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "mediapipe<0.11"
```

`requirements.txt` already pins `mediapipe==0.10.21`; the third command is an
explicit belt-and-braces guard so a future loosening of the pin cannot silently
pull in a `solutions`-free build.

## 5. Verify the install

```powershell
python -c "import mediapipe as mp; print(mp.__version__); mp.solutions.face_mesh; print('solutions OK')"
```

Expected:

```
0.10.21
solutions OK
```

Then check the detector itself constructs:

```powershell
python -c "from mediapipe_detector import MediaPipeEyeDetector; d=MediaPipeEyeDetector(); print('detector OK'); d.close()"
```

## 6. Run the app

```powershell
python main.py
```

---

## Resolved versions in the verified environment

Recorded from the environment this project was last verified in.

| Package | Version |
|---|---|
| **Python** | **3.12.10** |
| **mediapipe** | **0.10.21** |
| numpy | 1.26.4 |
| opencv-python | 4.11.0.86 |
| tensorflow | 2.19.1 |
| protobuf | 4.25.9 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |
| librosa | 0.11.0 |
| sounddevice | 0.5.5 |
| Pillow | 12.3.0 |
| pandas | 3.0.5 |
| scipy | 1.17.1 |
| openpyxl | 3.1.5 |
| flask | 3.1.3 |

Freeze your own for comparison with:

```powershell
python -m pip freeze > frozen.txt
```

---

## Hardware needed

| Test | Needs |
|---|---|
| Blink (60 s) | webcam |
| Gaze (10 trials) | webcam + a participant who can follow the on-screen target |
| Voice | microphone, **or** `python voice_dimentia.py --audio input.wav --plot` for a file |

The gaze test opens a full-screen stimulus window. Press **ESC** to abort it.

---

## Troubleshooting

**`UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f9e0'`**
`main.py` prints emoji and your console is on the legacy cp1252 code page. Either
use Windows Terminal, or set UTF-8 for the session:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python main.py
```

**`Could not open webcam!`**
Another application holds the camera. Close Teams/Zoom/Camera and retry. The app
tries `CAP_DSHOW`, then the default backend, then camera index 1.

**Scores differ from a colleague's on the same recording**
Check whether `blink_rf_model.pkl` or `gaze_rf_model.pkl` is present in the
repository root. Either file silently replaces the rule-based eye/gaze score with
model output in `gui_app.populate_results()`. Neither is committed.
`dev/verify_scores.py` aborts if it finds one.

**Verify scoring is unchanged after edits**

```powershell
python dev\verify_scores.py --out my_scores.json
```

Expected on an unmodified scoring path: blink **36.6667** (MILD), gaze **20.4**
(MILD), voice **55.0** (MODERATE), overall **39.9** (MILD).

---

## Notes

- `.venv312/` is gitignored. Never commit it.
- Python 3.13+ will not work until the detector is ported to the MediaPipe
  `tasks` API. See "Known limitations" in `README.md`.
