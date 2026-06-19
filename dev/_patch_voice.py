"""
Regex-based patch for voice_dimentia.py — handles CRLF line endings.
Portable: resolves voice_dimentia.py relative to this script's directory
so it works on any machine after cloning the repository.
"""
import re, ast, os

# Portable path — works everywhere, no hard-coded user directory
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voice_dimentia.py')
src  = open(path, 'r', encoding='utf-8').read()

results = {}

# ── PATCH 1: Jitter — insert 15s clip before try: block ──────────────────
# Matches: "f0 = librosa.yin(signal," anywhere in _estimate_jitter
if 'librosa.yin(signal,' in src and '15 * self.sample_rate' not in src:
    src = src.replace(
        'librosa.yin(signal,',
        'librosa.yin(sig,', 1)
    # Insert the truncation line before "try:"  inside _estimate_jitter
    src = re.sub(
        r'(def _estimate_jitter\(self, signal\):.*?""".*?""")',
        r'\1\n        sig = signal[:int(15 * self.sample_rate)]  # OPT: 15s clip',
        src, count=1, flags=re.DOTALL)
    results['Jitter 15s'] = 'PATCHED'
elif '15 * self.sample_rate' in src:
    results['Jitter 15s'] = 'already done'
else:
    results['Jitter 15s'] = 'WARN: pattern not found'

# ── PATCH 2: Shimmer — insert 10s clip at top of method ──────────────────
if '10 * self.sample_rate' not in src:
    src = re.sub(
        r'(def _estimate_shimmer\(self, signal\):(?:\r?\n        """.*?""")?(?:\r?\n))',
        r'\g<1>        signal = signal[:int(10 * self.sample_rate)]  # OPT: 10s clip\n',
        src, count=1, flags=re.DOTALL)
    results['Shimmer 10s'] = 'PATCHED'
else:
    results['Shimmer 10s'] = 'already done'

# ── PATCH 3: n_mfcc 42 → 30 ──────────────────────────────────────────────
if 'n_mfcc=42' in src:
    src = src.replace('n_mfcc=42', 'n_mfcc=30', 1)
    results['n_mfcc=30'] = 'PATCHED'
elif 'n_mfcc=30' in src:
    results['n_mfcc=30'] = 'already done'
else:
    results['n_mfcc=30'] = 'WARN: not found'

# ── PATCH 4: Vectorise clipping loop ─────────────────────────────────────
if 'for i, fn in enumerate(FEATURE_ORDER)' in src:
    src = re.sub(
        r'        # Clip to training range\r?\n'
        r'        for i, fn in enumerate\(FEATURE_ORDER\):\r?\n'
        r'            _, _, lo, hi = TRAINING_STATS\[fn\]\r?\n'
        r'            raw\[0, i\] = np\.clip\(raw\[0, i\], lo, hi\)',
        '        # OPT: vectorised clip\n'
        '        _lo = np.array([TRAINING_STATS[fn][2] for fn in FEATURE_ORDER])\n'
        '        _hi = np.array([TRAINING_STATS[fn][3] for fn in FEATURE_ORDER])\n'
        '        raw = np.clip(raw, _lo, _hi)',
        src, count=1)
    results['Vectorised clip'] = 'PATCHED'
elif '_clip_lo' in src or '_lo = np.array' in src:
    results['Vectorised clip'] = 'already done'
else:
    results['Vectorised clip'] = 'WARN: loop not found'

# ── Save and verify ───────────────────────────────────────────────────────
open(path, 'w', encoding='utf-8').write(src)

for name, status in results.items():
    print(f'  [{status}] {name}')

# Final check
content = open(path, encoding='utf-8').read()
checks = {
    'np.correlate REMOVED': 'np.correlate' not in content,
    'FFT autocorr present': 'np.fft.rfft' in content,
    'jitter 15s':           '15 * self.sample_rate' in content,
    'shimmer 10s':          '10 * self.sample_rate' in content,
    'n_mfcc=30':            'n_mfcc=30' in content,
    'n_mfcc=42 gone':       'n_mfcc=42' not in content,
    'no for-loop clip':     'for i, fn in enumerate(FEATURE_ORDER)' not in content,
}
print()
for name, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')

try:
    ast.parse(content)
    print('\n  Syntax: OK')
except SyntaxError as e:
    print(f'\n  Syntax ERROR: {e}')
