# 🧠 Dementia Detection System - Python GUI Version

## Quick Start Guide

### Running the Python GUI Application

1. **Install Dependencies** (if not already installed):
```bash
cd C:\Users\LEGION\Downloads\archive
pip install -r requirements.txt
```

2. **Launch the GUI**:
```bash
python gui_app.py
```

3. **Use the Application**:
   - Click **"▶ START DETECTION"** to begin webcam analysis
   - Watch real-time metrics update every second
   - Click **"⏹ STOP DETECTION"** to pause
   - Click **"💾 EXPORT DATA"** to save session data to CSV

---

## Features

### ✅ All Features from HTML Version Included:

| Feature | Status |
|---------|--------|
| Live video feed with eye tracking | ✅ |
| Start/Stop/Export buttons | ✅ |
| Blink Rate (blinks/min) | ✅ |
| Total Blinks count | ✅ |
| Eye Openness (EAR) | ✅ |
| Session Time (seconds) | ✅ |
| Risk Level indicator with color coding | ✅ |
| Risk Factors list | ✅ |
| Eye Openness Over Time graph | ✅ |
| Blink Pattern graph | ✅ |

---

## GUI Layout

```
┌─────────────────────────────────────────────────────────────┐
│          🧠 Dementia Detection System                       │
│     Real-time Eye Tracking & Blink Rate Analysis            │
├─────────────────────────────────────────────────────────────┤
│  System Ready - Click Start to Begin                        │
├──────────────────────────┬──────────────────────────────────┤
│  📹 Live Video Feed      │  📊 Real-time Metrics            │
│  ┌────────────────────┐  │  ┌──────────────────────────┐   │
│  │                    │  │  │   Risk Level: LOW        │   │
│  │   Webcam Display   │  │  │        0%                │   │
│  │   with Eye Overlay │  │  └──────────────────────────┘   │
│  │                    │  │  ┌──────┬──────┬──────┬──────┐  │
│  └────────────────────┘  │  │Blink │Total │ Eye  │Session│ │
│  [▶ START] [⏹ STOP]     │  │Rate  │Blinks│Open  │Time  │  │
│  [💾 EXPORT DATA]        │  └──────┴──────┴──────┴──────┘  │
│                          │  ⚠️ Risk Factors Detected       │
│                          │  • Low blink rate (11.4 bpm)    │
├──────────────────────────┴──────────────────────────────────┤
│  📈 Analysis Graphs                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ Eye Openness        │  │ Blink Pattern       │          │
│  │ Over Time (Line)    │  │ (Bar Chart)         │          │
│  └─────────────────────┘  └─────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Advantages Over HTML Version

| Aspect | HTML Version | Python GUI Version |
|--------|--------------|-------------------|
| **Installation** | Requires Flask server | No server needed |
| **Performance** | Network overhead | Direct processing |
| **Startup** | Open browser + server | Single command |
| **Portability** | Need web browser | Standalone app |
| **Resource Usage** | Browser + Server | Just Python |

---

## Files Created

### New Files:
- **`gui_app.py`** - Main GUI application (Tkinter)
- **`gui_adapter.py`** - Integration layer between GUI and analyzer

### Modified Files:
- **`requirements.txt`** - Added comments for GUI dependencies

### Unchanged (Still Used):
- **`dementia_analyzer.py`** - Core analysis logic
- **`blink_detection.py`** - Blink detection algorithms
- **`models/`** - Trained ML models

---

## Troubleshooting

### Issue: "No module named 'tkinter'"
**Solution**: Tkinter comes with Python. If missing, reinstall Python with Tkinter support.

### Issue: Webcam not opening
**Solution**: 
1. Check if another app is using the webcam
2. Try changing camera index in `gui_adapter.py` (line with `cv2.VideoCapture(0)`)

### Issue: Graphs not updating
**Solution**: Ensure matplotlib is installed: `pip install matplotlib`

### Issue: "No face detected"
**Solution**:
- Ensure good lighting
- Position face directly in front of camera
- Move closer to camera

---

## Comparison: HTML vs Python GUI

### To run HTML version:
```bash
python app.py
# Then open browser to http://localhost:5000
```

### To run Python GUI version:
```bash
python gui_app.py
# GUI window opens directly
```

**Both versions work independently!** You can keep both and use whichever you prefer.

---

## Next Steps

1. ✅ Test the GUI: `python gui_app.py`
2. ✅ Compare with HTML version
3. ✅ Choose your preferred interface
4. ✅ Delete unnecessary files if desired

---

**Made with ❤️ - Now with native Python GUI!**
