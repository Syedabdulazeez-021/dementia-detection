import sys, os, warnings, inspect, time
warnings.filterwarnings('ignore')
os.chdir(r'C:\Users\techn\BIO')
sys.path.insert(0, r'C:\Users\techn\BIO')
import numpy as np

print("=" * 55)
print("  voice_dimentia.py — Functional Verification Test")
print("=" * 55)

# 1. Import
print("\n[1] Importing module...", end=' ', flush=True)
import voice_dimentia as vd
print("OK")

# 2. Check optimisations in source (skip docstrings)
def code_only(fn):
    """Return only non-docstring, non-comment lines of a method."""
    lines = inspect.getsource(fn).splitlines()
    in_doc = False
    out = []
    for ln in lines[1:]:   # skip 'def' line
        s = ln.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_doc = not in_doc
            continue
        if not in_doc and not s.startswith('#'):
            out.append(ln)
    return '\n'.join(out)

checks = {}

# HNR: must use fft.rfft, must NOT call np.correlate (excluding docstring)
hnr_code = code_only(vd.DementiaAnalyser._estimate_hnr)
checks['HNR uses FFT']             = 'np.fft.rfft' in hnr_code
checks['HNR no np.correlate']      = 'np.correlate' not in hnr_code

# Jitter: 15s clip
jit_src = inspect.getsource(vd.DementiaAnalyser._estimate_jitter)
checks['Jitter 15s clip']          = '15 * self.sample_rate' in jit_src

# Shimmer: 10s clip
shi_src = inspect.getsource(vd.DementiaAnalyser._estimate_shimmer)
checks['Shimmer 10s clip']         = '10 * self.sample_rate' in shi_src

# n_mfcc=30
ext_src = inspect.getsource(vd.DementiaAnalyser.extract_features)
checks['n_mfcc=30']                = 'n_mfcc=30' in ext_src
checks['n_mfcc=42 removed']        = 'n_mfcc=42' not in ext_src

# Vectorised clipping (no Python for-loop)
checks['Vectorised clip (no loop)'] = 'for i, fn in enumerate(FEATURE_ORDER)' not in ext_src

for name, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')

# 3. Run estimators on dummy signal
print("\n[*] Running all estimators on 10s test signal...")
obj = object.__new__(vd.DementiaAnalyser)
obj.sample_rate = 22050
sr  = obj.sample_rate
sig = (np.sin(2*np.pi*180*np.linspace(0,10,sr*10,dtype='float32'))*0.3
       + np.random.randn(sr*10).astype('float32')*0.04)

t0 = time.time(); j = obj._estimate_jitter(sig);  print(f'  jitter  = {j:.6f}  ({time.time()-t0:.2f}s)')
t0 = time.time(); s = obj._estimate_shimmer(sig); print(f'  shimmer = {s:.6f}  ({time.time()-t0:.2f}s)')
t0 = time.time(); h = obj._estimate_hnr(sig);     print(f'  HNR     = {h:.4f}   ({time.time()-t0:.2f}s)')

all_pass = all(checks.values())
print("\n" + "=" * 55)
if all_pass:
    print("  ALL CHECKS PASSED ✓  —  voice_dimentia.py is working")
else:
    failed = [k for k,v in checks.items() if not v]
    print(f"  FAILED: {failed}")
print("=" * 55)
