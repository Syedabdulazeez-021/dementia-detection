"""
utils/audio_features.py
=======================
Shared audio feature extraction helpers used by the dementia detection pipeline.
All functions are pure (no global state) and return numpy arrays or plain scalars.

Usage:
    from utils.audio_features import estimate_speech_rate, estimate_pitch_variation
"""

import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ── Optional dependency ───────────────────────────────────────────────────────
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def estimate_speech_rate(signal: np.ndarray, sr: int = 22050) -> float:
    """
    Estimate speech rate in syllables per second using energy-burst counting.

    The algorithm counts low-to-high energy transitions in the RMS envelope,
    which correspond approximately to syllable onsets.

    Parameters
    ----------
    signal : np.ndarray  — mono audio, float32
    sr     : int         — sample rate (default 22050 Hz)

    Returns
    -------
    float  — estimated syllables per second (0.0 on failure)
    """
    if not LIBROSA_AVAILABLE:
        return 0.0

    try:
        hop = 512
        rms = librosa.feature.rms(y=signal, frame_length=2048, hop_length=hop)[0]
        threshold = np.mean(rms) * 0.5
        above = rms > threshold
        onsets = np.where(np.diff(above.astype(int)) == 1)[0]
        duration_s = len(signal) / sr
        return float(len(onsets) / max(duration_s, 1e-6))
    except Exception:
        return 0.0


def estimate_pitch_variation(signal: np.ndarray, sr: int = 22050,
                              clip_secs: float = 15.0) -> float:
    """
    Estimate pitch (F0) variability as the standard deviation of voiced frames.

    Parameters
    ----------
    signal     : np.ndarray  — mono audio, float32
    sr         : int         — sample rate (default 22050 Hz)
    clip_secs  : float       — only analyse the first N seconds (speed optimisation)

    Returns
    -------
    float  — std of voiced F0 in Hz (0.0 on failure or no voiced frames)
    """
    if not LIBROSA_AVAILABLE:
        return 0.0

    try:
        seg = signal[:int(clip_secs * sr)]
        f0 = librosa.yin(seg,
                         fmin=librosa.note_to_hz('C2'),
                         fmax=librosa.note_to_hz('C7'),
                         sr=sr,
                         frame_length=2048, hop_length=512)
        voiced = f0[f0 > 65]
        return float(np.std(voiced)) if len(voiced) > 3 else 0.0
    except Exception:
        return 0.0


def compute_spectral_centroid(signal: np.ndarray,
                               sr: int = 22050) -> np.ndarray:
    """
    Compute the spectral centroid over time (vocal brightness measure).

    Parameters
    ----------
    signal : np.ndarray  — mono audio, float32
    sr     : int         — sample rate (default 22050 Hz)

    Returns
    -------
    np.ndarray  — spectral centroid values in Hz, shape (n_frames,)
                  Empty array on failure.
    """
    if not LIBROSA_AVAILABLE:
        return np.array([])

    try:
        return librosa.feature.spectral_centroid(y=signal, sr=sr,
                                                 hop_length=512)[0]
    except Exception:
        return np.array([])


def rolling_rms(signal: np.ndarray, sr: int = 22050,
                frame_length: int = 2048, hop: int = 512) -> np.ndarray:
    """
    Compute the RMS energy over time using librosa.

    Parameters
    ----------
    signal       : np.ndarray  — mono audio, float32
    sr           : int         — sample rate (default 22050 Hz)
    frame_length : int         — FFT frame size
    hop          : int         — hop length in samples

    Returns
    -------
    np.ndarray  — RMS values, shape (n_frames,)
    """
    if not LIBROSA_AVAILABLE:
        return np.array([])

    try:
        return librosa.feature.rms(y=signal, frame_length=frame_length,
                                   hop_length=hop)[0]
    except Exception:
        return np.array([])


def compute_mfcc_heatmap(signal: np.ndarray, sr: int = 22050,
                          n_mfcc: int = 13) -> np.ndarray:
    """
    Compute the MFCC matrix suitable for a heatmap display.

    Parameters
    ----------
    signal : np.ndarray  — mono audio, float32
    sr     : int         — sample rate
    n_mfcc : int         — number of MFCC coefficients (default 13)

    Returns
    -------
    np.ndarray  — shape (n_mfcc, n_frames), float32
                  Empty array (0, 0) on failure.
    """
    if not LIBROSA_AVAILABLE:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        return librosa.feature.mfcc(y=signal, sr=sr,
                                    n_fft=2048, hop_length=512,
                                    n_mfcc=n_mfcc)
    except Exception:
        return np.zeros((0, 0), dtype=np.float32)
