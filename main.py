"""
main.py — Universal entry point for the BIO Dementia Detection System
=======================================================================
Setup:
    git clone <repository>
    pip install -r requirements.txt
    python main.py

No hard-coded paths. Works on any machine with Python 3.8+.
"""

import sys
import os

# ── Ensure the project root is on sys.path so all imports resolve ────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Friendly startup banner ──────────────────────────────────────────────────
print("=" * 65)
print("  🧠  Cognitive Risk Assessment")
print("=" * 65)
print(f"  Python : {sys.version.split()[0]}")
print(f"  Root   : {ROOT}")
print("=" * 65)

# ── Check required model files ───────────────────────────────────────────────
_VOICE_MODEL  = os.path.join(ROOT, "dementia_rf_model.pkl")
_VOICE_SCALER = os.path.join(ROOT, "scaler.pkl")
_EYE_MODEL    = os.path.join(ROOT, "models", "eye_classifier_best.h5")

missing = []
for f in [_VOICE_MODEL, _VOICE_SCALER, _EYE_MODEL]:
    if not os.path.exists(f):
        missing.append(os.path.relpath(f, ROOT))

if missing:
    print("\n  [WARNING] The following model files were not found:")
    for m in missing:
        print(f"    • {m}")
    print("\n  Voice/Eye analysis may be limited without these files.")
    print("  See README.md for instructions on obtaining them.\n")

# ── Launch the GUI ───────────────────────────────────────────────────────────
try:
    from gui_app import main as gui_main
    gui_main()
except ImportError as e:
    print(f"\n[ERROR] Failed to import GUI: {e}")
    print("Please run:  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    import traceback
    print(f"\n[ERROR] Unexpected error: {e}")
    traceback.print_exc()
    sys.exit(1)
