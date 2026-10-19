"""Read synchronized WESAD pickles without modifying source recordings.

Only load locally trusted pickle files: Python pickle can execute code.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

RATES = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
LABEL_RATE = 700
LABEL_NAMES = {0: "undefined", 1: "baseline", 2: "stress", 3: "amusement",
               4: "meditation", 5: "ignored", 6: "ignored", 7: "ignored"}
EXPECTED_SUBJECTS = {f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]}


@dataclass
class Recording:
    subject: str
    wrist: dict[str, np.ndarray]
    labels: np.ndarray

    @property
    def duration(self) -> float:
        return len(self.labels) / LABEL_RATE


def discover(root: Path) -> list[Path]:
    paths = sorted(root.glob("**/S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    if not paths:
        raise FileNotFoundError(f"No subject pickles under {root}")
    ids = [p.stem for p in paths]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subject recordings")
    missing = EXPECTED_SUBJECTS - set(ids)
    if missing:
        raise ValueError(f"Missing WESAD subjects: {sorted(missing)}")
    return paths


def validate(recording: Recording) -> None:
    if recording.labels.ndim != 1 or not len(recording.labels):
        raise ValueError("Labels must be a nonempty vector")
    if not np.isin(recording.labels, list(LABEL_NAMES)).all():
        raise ValueError("Unknown label values")
    for sensor, rate in RATES.items():
        samples = recording.wrist[sensor]
        channels = 3 if sensor == "ACC" else 1
        if samples.ndim != 2 or samples.shape[1] != channels:
            raise ValueError(f"{sensor}: invalid shape {samples.shape}")
        if not np.isfinite(samples).all():
            raise ValueError(f"{sensor}: nonfinite samples")
        if abs(len(samples) / rate - recording.duration) > 1 / rate + 1 / LABEL_RATE:
            raise ValueError(f"{sensor}: duration differs from labels")


def load_recording(path: Path) -> Recording:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    if str(data["subject"]) != path.stem:
        raise ValueError("Subject identity mismatch")
    recording = Recording(path.stem, {
        sensor: np.asarray(data["signal"]["wrist"][sensor]) for sensor in RATES
    }, np.asarray(data["label"]))
    del data
    validate(recording)
    return recording
