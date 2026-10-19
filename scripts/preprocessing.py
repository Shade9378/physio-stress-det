"""Prepare full-sensor or ACC-only app features from local WESAD recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.pycache_prefix = str(Path(__file__).resolve().parents[1] / "results/cache/python")

import numpy as np
import pandas as pd

from physio.processing.dataset import discover, load_recording
from physio.processing.windows import WindowConfig, eligible_windows, slice_window
from physio.processing.sensor_contracts import SENSORS, extract_features, feature_names, validate_table
from physio.processing.live_features import names_for, ppg_features, temperature_features, eda_features
from scipy import signal
from physio.processing.acceleration import FEATURE_NAMES, magnitude_bins, motion_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/WESAD/WESAD"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sensors", choices=["full", "acc", "live"], default="full")
    parser.add_argument("--from-features", type=Path,
                        help="Explicitly reuse a matching feature CSV")
    args = parser.parse_args()
    args.output = args.output or Path({"full": "results/preprocessing/multisensor", "acc": "results/preprocessing/watch", "live": "results/preprocessing/live_2025"}[args.sensors])
    names = names_for(["ACC", "BVP", "EDA", "TEMP"]) if args.sensors == "live" else (feature_names(list(SENSORS)) if args.sensors == "full" else FEATURE_NAMES)
    target = args.output / "features.csv"
    if target.exists():
        raise FileExistsError(f"{target} already exists; choose a new --output")
    config = json.loads(Path("configs/wear.json").read_text())
    if config["window_seconds"] != 60 or config["bin_hz"] != 16:
        raise ValueError("The native app requires 60-second, 16-Hz windows")
    provenance = {"sensor_contract": args.sensors, "window_seconds": 60, "stride_seconds": config["stride_seconds"],
                  "feature_names": names}
    if args.from_features:
        manifest = args.from_features.parent / "feature_manifest.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            for key, expected in provenance.items():
                if previous.get(key) != expected:
                    raise ValueError(f"Incompatible cached feature contract: {key}")
        frame = pd.read_csv(args.from_features)
        provenance.update({"source": str(args.from_features),
                           "source_sha256": hashlib.sha256(args.from_features.read_bytes()).hexdigest(),
                           "mode": "explicit_existing_feature_reuse"})
    else:
        rows = []
        sources = []
        window = WindowConfig(seconds=60, stride=config["stride_seconds"])
        for path in discover(args.data_root):
            recording = load_recording(path)
            bins = magnitude_bins(recording.wrist["ACC"])
            if args.sensors == "live":
                # Anti-alias each continuous stream before slicing; avoid artificial window-edge transients.
                live_ppg = signal.resample_poly(recording.wrist["BVP"][:, 0], 25, 64)
                live_temp = signal.resample_poly(recording.wrist["TEMP"][:, 0], 1, 4)
                live_eda = signal.resample_poly(recording.wrist["EDA"][:, 0], 1, 4)
            for start, label in eligible_windows(recording.labels, window):
                values = (extract_features(slice_window(recording, start, 60)) if args.sensors == "full"
                          else dict(zip(FEATURE_NAMES, motion_features(bins[start * 16:(start + 60) * 16]))))
                if args.sensors == "live":
                    ppg = live_ppg[start*25:(start+60)*25]
                    temp = live_temp[start:start+60]
                    extra = np.r_[ppg_features(ppg), eda_features(live_eda[start:start+60]), temperature_features(temp)]
                    values.update(dict(zip(names[6:], extra)))
                rows.append({"subject": recording.subject, "start_seconds": start,
                             "target": label, **values})
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            sources.append({"path": str(path), "sha256": digest})
            print(f"Prepared {recording.subject}", flush=True)
            del recording, bins
        frame = pd.DataFrame(rows)
        provenance.update({"sources": sources, "mode": "local_wesad_extraction"})
    validate_table(frame, names)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    provenance.update({"rows": len(frame), "subjects": sorted(frame.subject.unique()),
                       "features_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (args.output / "feature_manifest.json").write_text(json.dumps(provenance, indent=2))
    print(f"Saved {len(frame)} windows from {frame.subject.nunique()} subjects: {target}")


if __name__ == "__main__":
    main()
