"""Restore the validated supervised MLP and held-out replay assets without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.pycache_prefix = str(Path(__file__).resolve().parents[1] / "results/cache/python")
from physio.deployment.checkpoints import load_checkpoint
import numpy as np
import onnxruntime as ort
import pandas as pd
from physio.processing.dataset import discover, load_recording
from physio.processing.acceleration import FEATURE_NAMES, magnitude_bins, motion_features
from physio.deployment.replay_assets import export_sensor_bundle, export_local_replay
from physio.deployment.live_assets import export_live_assets, CASES as LIVE_CASES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--replay-only", action="store_true", help="Prepare local replay from installed release models without checkpoints")
parser.add_argument("--live-only", action="store_true", help="Export completed native watch-rate models only")
parser.add_argument("--model-dir", type=Path, help="Completed train_model.py output; omit to use available trained assets")
parser.add_argument("--features", type=Path, help="Override the matching six-feature CSV")
parser.add_argument("--data-root", type=Path, default=Path("data/WESAD/WESAD"))
parser.add_argument("--assets-dir", type=Path, default=Path("results/app/assets/generated"))
parser.add_argument("--export-dir", type=Path, default=Path("results/app/export"))
parser.add_argument("--sensor-models", type=Path, default=Path("results/experiments/wear/missing_modalities_v1"),
                    help="Validated full/subset model collection to package for prepared replay")
args = parser.parse_args()
if args.replay_only:
    evidence = export_local_replay(args.assets_dir, args.data_root)
    print(f"Prepared {evidence['windows']} local replay windows; models unchanged")
    sys.exit(0)
if args.live_only:
    evidence = export_live_assets(args.assets_dir, args.export_dir, args.data_root)
    print(f"Exported {len(evidence['models'])} native live sensor models")
    sys.exit(0)
sensor_override = None
if args.model_dir and (args.model_dir / "training_manifest.json").exists():
    manifest = json.loads((args.model_dir / "training_manifest.json").read_text())
    if len(manifest["feature_names"]) > 6:
        marker = args.model_dir / "validated.json"
        if not marker.exists() or json.loads(marker.read_text()).get("status") != "validated":
            raise ValueError("Complete training first: export requires a validated model directory")
        sensor_override, args.model_dir = args.model_dir, None
trained = Path("results/training/watch_mlp")
completed = trained / "validated.json"
legacy = Path("results/experiments/wear/distillation_v1")
root = args.model_dir or (trained if completed.exists() and
    json.loads(completed.read_text()).get("status") == "validated" else legacy)
marker = root / "validated.json"
if not marker.exists() or json.loads(marker.read_text()).get("status") != "validated":
    raise ValueError("Complete training first: export requires a validated model directory")
config = json.loads(Path("configs/wear.json").read_text())
assert config["model_family"] == "mlp"
subject, seed = config["deployment_held_out"], config["deployment_seed"]
source = root / subject / f"mlp_supervised_{seed}.onnx"
fitted_model = load_checkpoint(root / subject / f"mlp_supervised_{seed}.joblib")
assert fitted_model.candidate.get("alpha", 0.) == 0. and fitted_model.seed == seed
assert sum(p.numel() for p in fitted_model.model.parameters()) == 129
audit = json.loads((root / subject / "splits.json").read_text())
assert subject not in audit["outer_train"]
assets = args.assets_dir
deployment = args.export_dir
deployment.mkdir(parents=True, exist_ok=True)
# Retain the previous packaged assets once, rather than overwriting experiment evidence.
backup = deployment / "baseline_assets"
if assets.exists() and not backup.exists():
    shutil.copytree(assets, backup)
assets.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, assets / "stress.onnx")
baseline = Path("results/experiments/wear/phase2/models") / f"acc_{subject}.onnx"
if baseline.exists():
    shutil.copyfile(baseline, assets / "baseline_logistic.onnx")
options = ort.SessionOptions()
options.intra_op_num_threads = 1
session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
recording = load_recording(next(p for p in discover(args.data_root) if p.stem == subject))
bins = magnitude_bins(recording.wrist["ACC"])
feature_path = args.features or (root / "features.csv" if (root / "features.csv").exists()
                                 else Path("results/experiments/wear/phase2/features.csv"))
frame = pd.read_csv(feature_path)
frame = frame[frame.subject == subject]
segments = []
for target in [0, 1]:
    starts = set(frame[frame.target == target].start_seconds.astype(int))
    start = next(s for s in sorted(starts) if all(s + t in starts for t in range(0, 71, 10)))
    segments.append({"subject": subject, "start_seconds": start, "target": target,
                     "magnitude_g": bins[start*16:(start+130)*16].tolist(),
                     "acc_raw": recording.wrist["ACC"][start*32:(start+130)*32].tolist()})
vectors = []
for segment in segments:
    values = np.asarray(segment["magnitude_g"])
    for end in range(960, len(values) + 1, 160):
        features = motion_features(values[end-960:end])
        probability = float(fitted_model.predict(features[None])[0])
        converted = float(session.run(None, {"features": features.astype(np.float32)[None]})[0][0, 0])
        assert abs(probability - converted) < 5e-5
        vectors.append({"bins": values[end-960:end].tolist(), "features": features.tolist(),
                        "probability": probability, "target": segment["target"]})
(assets / "replay.json").write_text(json.dumps({"bin_hz": 16, "segments": segments}))
(assets / "parity.json").write_text(json.dumps(vectors))
evaluation = []
for row in frame.itertuples():
    features = np.array([getattr(row, name) for name in FEATURE_NAMES])
    evaluation.append({"start_seconds": row.start_seconds, "target": row.target,
        "features": features.tolist(), "probability": float(fitted_model.predict(features[None])[0])})
(assets / "evaluation.json").write_text(json.dumps(evaluation))
metrics = pd.read_csv(root / "fold_metrics.csv")
metrics = metrics[metrics.model == "mlp_supervised"]
seed_metrics = metrics[metrics.seed == seed]
score_names = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]


def summary(table: pd.DataFrame) -> dict:
    result = {}
    for name in score_names:
        result[name + "_mean"] = float(table[name].mean())
        result[name + "_std"] = float(table[name].std())
    return result


model_hash = hashlib.sha256(source.read_bytes()).hexdigest()
model_id = config["model_id"] if root == legacy else f"acc_mlp_{subject.lower()}_seed{seed}_{model_hash[:12]}"
metadata = {"schema": 2, **config, "model_id": model_id, "train_subjects": audit["outer_train"],
    "held_out_subject": subject, "uncalibrated": True, "evaluation": [summary(seed_metrics)],
    "research_three_seed_evaluation": summary(metrics.groupby("subject")[score_names].mean()),
    "held_out_subject_metrics": seed_metrics[seed_metrics.subject == subject][score_names].iloc[0].to_dict(),
    "architecture": "6 -> 16 ReLU -> 1 sigmoid; training scaler folded into ONNX",
    "parameters": 129, "model_bytes": source.stat().st_size,
    "model_sha256": model_hash,
    "comparison_baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest() if baseline.exists() else None,
    "source_checkpoint": str(root / subject / f"mlp_supervised_{seed}.joblib"),
    "input_contract": "60 seconds complete mean-magnitude 16Hz bins in g; includes gravity",
    "live_validation": "No real-world watch validation; protocol motion may confound stress",
    "incompatible_or_missing_acc": "Suppress classification; HR and missing modalities are not substitutes"}
(assets / "model.json").write_text(json.dumps(metadata, indent=2))
(deployment / "export.json").write_text(json.dumps({
    "status": "exported_pending_android_validation", "model": metadata,
    "replay_vectors": len(vectors), "held_out_evaluation_windows": len(evaluation)}, indent=2))
print(f"Restored supervised MLP: {len(vectors)} replay vectors, {len(evaluation)} held-out windows, "
      f"{metadata['model_bytes']} bytes; no retraining")
sensor_export = export_sensor_bundle(args.sensor_models, assets, deployment, recording, frame, sensor_override)
print(f"Packaged {len(sensor_export['models'])} sensor-specific MLPs; all four sensors enabled for replay")


if all((Path("results/training/live_sensor_subsets") / case / "validated.json").exists()
       for case in LIVE_CASES):
    export_live_assets(assets, deployment, args.data_root)
