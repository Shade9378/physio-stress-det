"""Window-local filters and features shared by offline and streaming inference."""
from dataclasses import dataclass
import warnings

import neurokit2 as nk
import numpy as np
from scipy import signal

from physio.processing.dataset import RATES, LABEL_RATE, Recording

SENSORS = tuple(RATES)
SEQUENCE_RATE = 16


@dataclass(frozen=True)
class WindowConfig:
    seconds: int = 60
    stride: int = 10

    def __post_init__(self) -> None:
        if self.seconds < 10 or self.stride <= 0:
            raise ValueError("Window must be >=10 seconds and stride positive")


def eligible_windows(labels: np.ndarray, config: WindowConfig) -> list[tuple[int, int]]:
    """Return grid-aligned, entirely single-condition windows, label mapped to 0/1."""
    width = config.seconds * LABEL_RATE
    stride = config.stride * LABEL_RATE
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    windows = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        label = int(labels[start])
        if label not in (1, 2):
            continue
        first = ((int(start) + stride - 1) // stride) * stride
        for offset in range(first, int(stop) - width + 1, stride):
            windows.append((offset // LABEL_RATE, label - 1))
    return windows


def slice_window(recording: Recording, start: float, seconds: int) -> dict[str, np.ndarray]:
    return {name: values[int(round(start*RATES[name])):int(round((start+seconds)*RATES[name]))]
            for name, values in recording.wrist.items()}


def filter_window(wrist: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for sensor, rate in RATES.items():
        values = np.asarray(wrist[sensor], dtype=np.float64)
        if values.ndim != 2 or len(values) < rate * 10 or not np.isfinite(values).all():
            raise ValueError(f"{sensor}: invalid/incomplete window")
        if sensor == "BVP":
            filtered = nk.ppg_clean(values[:, 0], sampling_rate=rate, method="elgendi")[:, None]
        else:
            cutoff = {"EDA": 0.5, "TEMP": 0.1, "ACC": 5}[sensor]
            sos = signal.butter(3, cutoff, fs=rate, output="sos")
            filtered = signal.sosfiltfilt(sos, values, axis=0)
        result[sensor] = filtered
    return result


def transform_window(wrist: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    """Filter only samples already available in this completed window.

    Zero-phase filters use both directions within the trailing window; never any
    future samples after its end. A new window is independently filtered.
    """
    cleaned = filter_window(wrist)
    features: dict[str, float] = {}
    channels = []
    for sensor, values in cleaned.items():
        rate = RATES[sensor]
        # 16 Hz Nyquist is compatible with the BVP filter's 8 Hz upper edge.
        resampled = signal.resample_poly(values, SEQUENCE_RATE, rate, axis=0)
        channels.append(resampled.T)
        for channel in range(values.shape[1]):
            x = values[:, channel]
            prefix = f"{sensor.lower()}_{channel}"
            for name, value in {
                "mean": np.mean(x), "std": np.std(x), "min": np.min(x),
                "max": np.max(x), "iqr": np.subtract(*np.percentile(x, [75, 25])),
                "slope": np.polyfit(np.arange(len(x))/rate, x, 1)[0],
                "diff_std": np.std(np.diff(x))}.items():
                features[f"{prefix}_{name}"] = float(value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peaks = nk.ppg_findpeaks(cleaned["BVP"][:, 0], sampling_rate=64)["PPG_Peaks"]
    intervals = np.diff(peaks) / 64
    intervals = intervals[(intervals >= 0.3) & (intervals <= 2)]
    features["bvp_heart_rate"] = float(60/np.median(intervals)) if len(intervals) >= 3 else np.nan
    features["bvp_rmssd"] = float(np.sqrt(np.mean(np.diff(intervals)**2))) if len(intervals) >= 3 else np.nan
    phasic = nk.eda_phasic(cleaned["EDA"][:, 0], sampling_rate=4, method="highpass")["EDA_Phasic"]
    peaks, _ = signal.find_peaks(phasic.to_numpy(), prominence=0.01, distance=4)
    features["eda_peak_count"] = float(len(peaks))
    features["eda_phasic_std"] = float(np.std(phasic))
    # Native ACC is in 1/64g. Motion flags inform quality; no label-dependent deletion.
    magnitude = np.linalg.norm(wrist["ACC"] / 64, axis=1)
    features["motion_dynamic_g"] = float(np.std(magnitude))
    features["bvp_valid_intervals"] = float(len(intervals))
    return np.concatenate(channels).astype(np.float32), features
