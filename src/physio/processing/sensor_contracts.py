"""Validated sensor contracts shared by app preparation, training and export.

The full contract reuses the existing WESAD subset study, including its original
NeuroKit2/SciPy preprocessing. Health Services BPM is never substituted for BVP.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from physio.processing.acceleration import FEATURE_NAMES, magnitude_bins, motion_features
from physio.processing.windows import transform_window

def sensor_features(full_names: list[str]) -> dict[str, list[str]]:
    return {
        "ACC": FEATURE_NAMES.copy(),
        "BVP": [name for name in full_names if name.startswith("bvp_")],
        "EDA": [name for name in full_names if name.startswith("eda_")],
        "TEMP": [name for name in full_names if name.startswith("temp_")],
        "HR_PROXY": ["bvp_heart_rate"],
    }


def features_for(sensors: list[str], full_names: list[str]) -> list[str]:
    mapping = sensor_features(full_names)
    if not sensors or any(sensor not in mapping for sensor in sensors):
        raise ValueError("An explicit nonempty known sensor set is required")
    if "BVP" in sensors and "HR_PROXY" in sensors:
        raise ValueError("Derived BPM is already included in BVP; do not duplicate it")
    return [name for sensor in sensors for name in mapping[sensor]]


SENSORS = ("ACC", "BVP", "EDA", "TEMP")
SENSOR_CASES = {"acc": ["ACC"]}
for size in (2, 3):
    for extra in combinations(SENSORS[1:], size - 1):
        SENSOR_CASES["acc_" + "_".join(s.lower() for s in extra)] = ["ACC", *extra]
SENSOR_CASES["full"] = list(SENSORS)
SENSOR_CASES["no_acc"] = list(SENSORS[1:])


def feature_names(sensors: list[str]) -> list[str]:
    physiological = [f"{s.lower()}_0_{stat}" for s in SENSORS[1:]
                     for stat in ("mean", "std", "min", "max", "iqr", "slope", "diff_std")]
    physiological += ["bvp_heart_rate", "bvp_rmssd", "eda_peak_count",
                      "eda_phasic_std", "bvp_valid_intervals"]
    return features_for(sensors, physiological)


def extract_features(wrist: dict[str, np.ndarray]) -> dict[str, float]:
    # Preserve the published 32-feature subset contract, not the original ACC axes.
    _, original = transform_window(wrist)
    bins = magnitude_bins(wrist["ACC"])
    motion = dict(zip(FEATURE_NAMES, motion_features(bins)))
    values = {**motion, **original}
    return {name: values[name] for name in feature_names(list(SENSORS))}


def validate_table(frame: pd.DataFrame, names: list[str]) -> None:
    if list(frame.columns) != ["subject", "start_seconds", "target", *names] or frame.empty:
        raise ValueError("Feature columns/order do not match the selected sensor contract")
    if not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite feature/window values; do not invent missing physiology")
    if set(frame.target.unique()) != {0, 1}:
        raise ValueError("Both baseline (0) and stress (1) are required")
    if (frame.start_seconds < 0).any() or (frame.start_seconds % 1 != 0).any():
        raise ValueError("Window starts must be nonnegative whole seconds")
    if frame.duplicated(["subject", "start_seconds"]).any():
        raise ValueError("Duplicate subject/window rows")
    if not frame.groupby("subject").target.nunique().eq(2).all():
        raise ValueError("Every subject must have both conditions")
