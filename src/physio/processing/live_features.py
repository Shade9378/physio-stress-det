"""2025 watch-rate feature contract; separate from the preserved WESAD models."""
import numpy as np
from scipy import signal
from physio.processing.acceleration import FEATURE_NAMES

PPG_NAMES = ["bvp_bpm", "bvp_rmssd_seconds", "bvp_interval_std_seconds",
             "bvp_valid_intervals", "bvp_normalized_iqr"]
EDA_NAMES = [f"eda_{s}" for s in ("mean", "std", "min", "max", "iqr", "slope", "diff_std")]
EDA_SOS = signal.butter(3, .2, fs=1, output="sos")
TEMP_NAMES = [f"temp_{s}" for s in ("mean", "std", "min", "max", "iqr", "slope", "diff_std")]
PPG_SOS = signal.butter(3, [.5, 8.], btype="bandpass", fs=25, output="sos")
TEMP_SOS = signal.butter(3, .1, fs=1, output="sos")


def names_for(sensors: list[str]) -> list[str]:
    if "ACC" not in sensors or set(sensors) - {"ACC", "BVP", "EDA", "TEMP"}:
        raise ValueError("2025 live contract requires ACC plus optional raw PPG, EDA and skin temperature")
    return FEATURE_NAMES + (PPG_NAMES if "BVP" in sensors else []) + (EDA_NAMES if "EDA" in sensors else []) + (TEMP_NAMES if "TEMP" in sensors else [])


def ppg_features(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=float).ravel()
    if x.shape != (1500,) or not np.isfinite(x).all():
        raise ValueError("Require a complete 60 s, 25 Hz raw PPG window")
    x = signal.sosfiltfilt(PPG_SOS, x)
    if x.std() < 1e-9:
        raise ValueError("Flat PPG")
    # Per-window normalization removes arbitrary ADC gain and offset, not sampling/domain shift.
    x = (x - x.mean()) / x.std()
    peaks = signal.find_peaks(x, distance=8, prominence=.3)[0]
    intervals = np.diff(peaks) / 25.
    intervals = intervals[(intervals >= .3) & (intervals <= 2.)]
    if len(intervals) < 3:
        raise ValueError("Insufficient valid pulse intervals")
    return np.array([60 / np.median(intervals), np.sqrt(np.mean(np.diff(intervals)**2)),
                     intervals.std(), len(intervals), np.percentile(x, 75)-np.percentile(x, 25)])


def temperature_features(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=float).ravel()
    if x.shape != (60,) or not np.isfinite(x).all():
        raise ValueError("Require a complete 60 s, 1 Hz skin-temperature window")
    x = signal.sosfiltfilt(TEMP_SOS, x)
    return np.array([x.mean(), x.std(), x.min(), x.max(),
                     np.percentile(x, 75)-np.percentile(x, 25),
                     np.polyfit(np.arange(60), x, 1)[0], np.diff(x).std()])


def eda_features(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=float).ravel()
    if x.shape != (60,) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Require a complete 60 s, 1 Hz EDA window in microsiemens")
    x = signal.sosfiltfilt(EDA_SOS, x)
    return np.array([x.mean(), x.std(), x.min(), x.max(),
                     np.percentile(x, 75)-np.percentile(x, 25),
                     np.polyfit(np.arange(60), x, 1)[0], np.diff(x).std()])


def filter_contract() -> dict:
    def bundle(sos: np.ndarray) -> dict:
        pad = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        return {"sos": sos.tolist(), "zi": signal.sosfilt_zi(sos).tolist(), "pad": int(pad)}
    return {"schema": 1, "PPG": bundle(PPG_SOS), "TEMP": bundle(TEMP_SOS), "EDA": bundle(EDA_SOS)}
