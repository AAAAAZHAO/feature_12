from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from scipy.signal import hilbert, lfilter
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for hilbert transform and filtering. Install scipy first."
    ) from exc


@dataclass
class PitchHilbertInstantResult:
    t0: np.ndarray
    logt0: np.ndarray
    f0: np.ndarray
    logf0: np.ndarray
    mnf0: float
    lmnf0: float
    lmaxf0: float
    lminf0: float
    lrnf0: float
    spf0: float


def _frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice a 1D signal into overlapping frames (copy-based for clarity)."""
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    num_frames = 1 + max(0, (len(signal) - frame_length) // hop_length)
    if num_frames <= 0:
        return np.empty((0, frame_length), dtype=signal.dtype)
    frames = np.zeros((num_frames, frame_length), dtype=signal.dtype)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i, :] = signal[start:end]
    return frames


def _overlap_add(frames: np.ndarray, hop_length: int) -> np.ndarray:
    """Overlap-add frames back into a 1D signal using sum of windows as weights."""
    if frames.size == 0:
        return np.array([], dtype=frames.dtype)
    num_frames, frame_length = frames.shape
    out_length = (num_frames - 1) * hop_length + frame_length
    output = np.zeros(out_length, dtype=frames.dtype)
    weight = np.zeros(out_length, dtype=frames.dtype)
    # Use Hann window for analysis-synthesis consistency
    window = np.hanning(frame_length).astype(frames.dtype)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        output[start:end] += frames[i, :] * window
        weight[start:end] += window
    # Avoid divide by zero
    nonzero = weight > 1e-12
    output[nonzero] = output[nonzero] / weight[nonzero]
    return output


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    """Biased autocorrelation for nonnegative lags."""
    if x.size == 0:
        return np.array([0.0], dtype=float)
    r = np.correlate(x, x, mode="full")
    mid = len(r) // 2
    return r[mid:]


def _levinson_durbin(r: np.ndarray, order: int) -> Tuple[np.ndarray, float]:
    """Levinson–Durbin recursion to solve Toeplitz system for LPC.

    Returns (a, err), where a[0] == 1.0 are LPC coefficients for A(z).
    """
    if r.ndim != 1:
        raise ValueError("autocorrelation r must be 1-D")
    if r[0] <= 0:
        a = np.zeros(order + 1, dtype=float)
        a[0] = 1.0
        return a, 0.0
    a = np.zeros(order + 1, dtype=float)
    e = r[0]
    a[0] = 1.0
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / e
        a_prev = a.copy()
        a[1:i] = a_prev[1:i] + k * a_prev[i - 1:0:-1]
        a[i] = k
        e *= (1.0 - k * k)
        if e <= 1e-12:
            # Numerical floor
            e = 1e-12
            break
    return a, float(e)


def compute_lpc_residual(
    signal: np.ndarray,
    fs: int = 8000,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    lpc_order: int = 10,
    preemph: float = 0.97,
) -> np.ndarray:
    """Compute LPC residual via per-frame inverse filtering with overlap-add.

    Parameters
    - signal: mono float32/float64 array in [-1, 1]
    - fs: sampling rate
    - frame_ms: analysis window length (ms)
    - hop_ms: hop size (ms)
    - lpc_order: LPC model order
    - preemph: pre-emphasis coefficient y[n] = x[n] - a*x[n-1]
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D mono")

    # Pre-emphasis
    sig = signal.astype(float, copy=False)
    sig = np.append(sig[0], sig[1:] - preemph * sig[:-1])

    frame_length = max(1, int(round(frame_ms * fs / 1000.0)))
    hop_length = max(1, int(round(hop_ms * fs / 1000.0)))

    frames = _frame_signal(sig, frame_length, hop_length)
    if frames.size == 0:
        return np.zeros_like(sig)

    window = np.hanning(frame_length).astype(float)
    windowed = frames * window[None, :]

    res_frames = np.zeros_like(windowed)
    for i in range(windowed.shape[0]):
        frame = windowed[i]
        # Autocorrelation up to order
        r = _autocorrelation(frame)
        if len(r) < lpc_order + 1:
            # Pad r with zeros if needed
            r = np.pad(r, (0, lpc_order + 1 - len(r)), mode="constant")
        a, _ = _levinson_durbin(r[: lpc_order + 1], lpc_order)
        # Inverse filter: apply A(z)
        res_frames[i] = lfilter(a, [1.0], frame)

    residual = _overlap_add(res_frames, hop_length)

    # Normalize residual to avoid extreme amplitudes
    max_abs = np.max(np.abs(residual)) if residual.size else 0.0
    if max_abs > 0:
        residual = residual / (1.01 * max_abs)

    return residual.astype(float)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Compute Hilbert envelope of a real signal."""
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    analytic = hilbert(x)
    env = np.abs(analytic)
    # Light smoothing to reduce spurious fine structure (5 ms moving average)
    win_ms = 5.0
    fs_est = 1.0  # unknown here; we'll infer from typical use. Skip if too short.
    # We cannot infer fs here; smoothing will be applied in tracker if needed.
    return env


def pitch_from_envelope_autocorr(
    envelope: np.ndarray,
    fs: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    fmin: float = 50.0,
    fmax: float = 400.0,
    clarity_threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate pitch track from Hilbert envelope using autocorrelation-per-frame.

    Returns (t0_cont_ms, f0_cont_hz) including zeros for unvoiced frames.
    """
    env = envelope.astype(float, copy=False)
    env = env / (1.01 * (np.max(env) + 1e-12))

    frame_length = max(1, int(round(frame_ms * fs / 1000.0)))
    hop_length = max(1, int(round(hop_ms * fs / 1000.0)))

    frames = _frame_signal(env, frame_length, hop_length)
    if frames.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    window = np.hanning(frame_length).astype(float)

    min_lag = max(1, int(round(fs / fmax)))
    max_lag = max(min_lag + 1, int(round(fs / fmin)))

    t0_list = []
    f0_list = []

    for i in range(frames.shape[0]):
        frame = frames[i] * window
        frame = frame - np.mean(frame)
        energy = np.sum(frame * frame)
        if energy < 1e-6:
            t0_list.append(0.0)
            f0_list.append(0.0)
            continue
        r = _autocorrelation(frame)
        r0 = r[0] if r[0] > 0 else 1e-12
        # Normalize autocorrelation (NACF)
        rn = r / r0
        search = rn[min_lag : max_lag + 1]
        if search.size == 0:
            t0_list.append(0.0)
            f0_list.append(0.0)
            continue
        peak_idx = int(np.argmax(search))
        peak_val = float(search[peak_idx])
        lag = min_lag + peak_idx
        if peak_val < clarity_threshold or lag <= 0:
            t0_list.append(0.0)
            f0_list.append(0.0)
            continue
        t0_ms = (lag / fs) * 1000.0
        f0_hz = fs / float(lag)
        t0_list.append(t0_ms)
        f0_list.append(f0_hz)

    t0_cont = np.asarray(t0_list, dtype=float)
    f0_cont = np.asarray(f0_list, dtype=float)
    return t0_cont, f0_cont


def pitch_hilbert_instant_parameters(
    sp1sig: np.ndarray,
    fs: int = 8000,
) -> PitchHilbertInstantResult:
    """Python equivalent of MATLAB pitch_hilbert_instant_parameters.

    Parameters
    - sp1sig: 1-D mono signal (float or int). Will be normalized.
    - fs: sampling rate (default 8000 Hz to match original)

    Returns
    - PitchHilbertInstantResult with arrays and summary metrics as in MATLAB code.
    """
    if sp1sig is None:
        raise ValueError("sp1sig cannot be None")
    x = np.asarray(sp1sig).astype(float).squeeze()
    if x.ndim != 1:
        raise ValueError("sp1sig must be 1-D mono")

    # Normalize similar to MATLAB: divide by 1.01*max(abs(x))
    max_abs = np.max(np.abs(x)) if x.size else 0.0
    if max_abs > 0:
        x = x / (1.01 * max_abs)

    # 1) LPC residual
    residual = compute_lpc_residual(x, fs=fs, frame_ms=20.0, hop_ms=10.0, lpc_order=10, preemph=0.97)

    # 2) Hilbert envelope of residual
    env = hilbert_envelope(residual)

    # 3) Pitch from envelope via autocorrelation on 30 ms frames and 10 ms hop
    t0_cont_ms, f0_cont_hz = pitch_from_envelope_autocorr(
        env, fs=fs, frame_ms=30.0, hop_ms=10.0, fmin=50.0, fmax=400.0, clarity_threshold=0.4
    )

    # 4) Mask out unvoiced (zero) and compute outputs matching MATLAB logic
    voiced_idx = f0_cont_hz > 0
    f0_vec = f0_cont_hz[voiced_idx]
    t0_vec = t0_cont_ms[voiced_idx]

    if f0_vec.size == 0:
        # No voiced frames detected
        empty = np.array([], dtype=float)
        nan = float("nan")
        return PitchHilbertInstantResult(
            t0=empty,
            logt0=empty,
            f0=empty,
            logf0=empty,
            mnf0=nan,
            lmnf0=nan,
            lmaxf0=nan,
            lminf0=nan,
            lrnf0=nan,
            spf0=nan,
        )

    logt0 = np.log10(t0_vec)
    logf0 = np.log10(f0_vec)

    mnf0 = float(np.mean(f0_vec))
    lmnf0 = float(np.log10(np.mean(f0_vec)))
    lmaxf0 = float(np.log10(np.max(f0_vec)))
    lminf0 = float(np.log10(np.min(f0_vec)))
    lrnf0 = float(np.log10(np.max(f0_vec) - np.min(f0_vec)))
    spf0 = float((f0_vec[-1] - f0_vec[0]) / len(f0_vec))

    return PitchHilbertInstantResult(
        t0=t0_vec,
        logt0=logt0,
        f0=f0_vec,
        logf0=logf0,
        mnf0=mnf0,
        lmnf0=lmnf0,
        lmaxf0=lmaxf0,
        lminf0=lminf0,
        lrnf0=lrnf0,
        spf0=spf0,
    )


def pitch_hilbert_instant_parameters_tuple(
    sp1sig: np.ndarray, fs: int = 8000
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Tuple-returning wrapper to match MATLAB output order.

    Returns (t0, logt0, f0, logf0, mnf0, lmnf0, lmaxf0, lminf0, lrnf0, spf0)
    """
    res = pitch_hilbert_instant_parameters(sp1sig, fs=fs)
    return (
        res.t0,
        res.logt0,
        res.f0,
        res.logf0,
        res.mnf0,
        res.lmnf0,
        res.lmaxf0,
        res.lminf0,
        res.lrnf0,
        res.spf0,
    )


__all__ = [
    "PitchHilbertInstantResult",
    "compute_lpc_residual",
    "hilbert_envelope",
    "pitch_from_envelope_autocorr",
    "pitch_hilbert_instant_parameters",
    "pitch_hilbert_instant_parameters_tuple",
]
