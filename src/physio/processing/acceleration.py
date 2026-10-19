"""Portable acceleration contract shared with the native Wear OS application."""
import numpy as np

FEATURE_NAMES = ["mean_g", "std_g", "range_g", "mean_absolute_delta_g",
                 "rms_delta_g", "fraction_dynamic"]
BIN_HZ = 16
WINDOW_BINS = 960


def magnitude_bins(acceleration: np.ndarray) -> np.ndarray:
    """WESAD 32 Hz 1/64-g axes -> 16 Hz mean magnitude, including gravity."""
    values = np.asarray(acceleration, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite acceleration axes")
    magnitude = np.linalg.norm(values / 64.0, axis=1)
    return magnitude[:len(magnitude) // 2 * 2].reshape(-1, 2).mean(axis=1)


def motion_features(bins: np.ndarray) -> np.ndarray:
    values = np.asarray(bins, dtype=np.float64)
    if values.shape != (WINDOW_BINS,) or not np.isfinite(values).all():
        raise ValueError("A complete 60-second 16 Hz window is required")
    delta = np.diff(values)
    return np.array([values.mean(), values.std(), np.ptp(values),
                     np.abs(delta).mean(), np.sqrt(np.mean(delta ** 2)),
                     np.mean(np.abs(values - 1.0) > 0.1)], dtype=np.float64)
