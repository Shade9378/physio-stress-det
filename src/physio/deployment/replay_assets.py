"""Package actual held-out multi-sensor models and prepared WESAD windows.

Only feature preparation runs on the host for full-sensor replay. The watch runs
the ONNX models locally. This does not claim native raw BVP/EDA/TEMP acquisition.
"""
import hashlib
import json
import re
from pathlib import Path
import shutil

from physio.deployment.checkpoints import load_checkpoint
import numpy as np
import onnxruntime as ort
import pandas as pd

from physio.processing.dataset import Recording
from physio.processing.windows import slice_window
from physio.processing.sensor_contracts import SENSOR_CASES, SENSORS, extract_features, feature_names


def export_sensor_bundle(source: Path, assets: Path, output: Path,
                         recording: Recording, windows: pd.DataFrame,
                         trained_override: Path | None = None) -> dict:
    if json.loads((source / "validated.json").read_text()).get("status") != "validated":
        raise ValueError("Sensor models require completed validation")
    subject, seed = recording.subject, 2024
    names = feature_names(list(SENSORS))
    rows = [extract_features(slice_window(recording, int(start), 60))
            for start in windows.start_seconds]
    frame = pd.DataFrame(rows, columns=names)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Incomplete physiological features; cannot package replay")
    held_out = windows.target.to_numpy()
    model_dir = assets / "sensor_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    models, predictions = [], {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    for case, sensors in SENSOR_CASES.items():
        folder = source / case / subject
        checkpoint = folder / f"mlp_{seed}.joblib"
        model = folder / f"mlp_{seed}.onnx"
        audit = json.loads((folder / "splits.json").read_text())
        selected_names = feature_names(sensors)
        assert audit["features"] == selected_names and subject not in audit["train_subjects"]
        # A newly completed standalone app-training run can replace its matching case.
        trained = Path("results/training/sensor_subsets") / case
        if trained_override is not None:
            override_manifest = json.loads((trained_override / "training_manifest.json").read_text())
            if override_manifest["sensors"] == sensors:
                trained = trained_override
        use_trained = case != "acc" and (trained / "validated.json").exists()
        if use_trained:
            assert json.loads((trained / "validated.json").read_text())["status"] == "validated"
            audit = json.loads((trained / subject / "splits.json").read_text())
            assert subject not in audit["outer_train"]
            manifest = json.loads((trained / "training_manifest.json").read_text())
            assert manifest["feature_names"] == selected_names
            checkpoint = trained / subject / f"mlp_supervised_{seed}.joblib"
            model = trained / subject / f"mlp_supervised_{seed}.onnx"
        if case == "acc":
            # Keep raw live acquisition and prepared replay on exactly the same model.
            model = assets / "stress.onnx"
            checkpoint = Path(json.loads((assets / "model.json").read_text())["source_checkpoint"])
        converted = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        values = frame[selected_names].to_numpy()
        probability = converted.run(None, {"features": values.astype(np.float32)})[0].ravel()
        fitted = load_checkpoint(checkpoint)
        reference = fitted.predict(values)
        error = float(np.max(np.abs(probability - reference)))
        if error >= 5e-5 or not np.array_equal(probability >= .5, reference >= .5):
            raise ValueError(f"Python/ONNX mismatch: {case}")
        destination = model_dir / f"{case}.onnx"
        shutil.copyfile(model, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        metrics_path = trained / "fold_metrics.csv" if use_trained else source / "fold_metrics.csv"
        scores = pd.read_csv(metrics_path)
        if "case" in scores:
            scores = scores[(scores["case"] == case) & (scores.family == "mlp")]
        averaged = scores.groupby("subject").macro_f1.mean()
        models.append({"case": case, "sensors": sensors, "feature_names": selected_names,
            "file": f"sensor_models/{case}.onnx", "model_sha256": digest,
            "model_id": f"{case}_mlp_s2_2024_{digest[:12]}", "model_family": "mlp",
            "input_count": len(selected_names), "window_seconds": 60,
            "held_out_subject": subject, "seed": seed, "model_bytes": destination.stat().st_size,
            "evaluation": [{"macro_f1_mean": float(averaged.mean()), "macro_f1_std": float(averaged.std())}],
            "parity_max_error": error, "live_compatible": case == "acc",
            "source_checkpoint": str(checkpoint), "source_onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "held_out_accuracy": float(np.mean((probability >= .5) == held_out))})
        predictions[case] = probability.tolist()
    # Replay the same two 130-second segments as the raw-ACC replay, at window boundaries.
    replay_source = json.loads((assets / "replay.json").read_text())
    segments = []
    for raw_segment in replay_source["segments"]:
        start = raw_segment["start_seconds"]
        selected = windows.start_seconds.between(start, start + 70)
        segment_rows = []
        for index in np.flatnonzero(selected.to_numpy()):
            segment_rows.append({"start_seconds": int(windows.start_seconds.iloc[index]),
                "target": int(held_out[index]), "features": rows[index],
                "available_sensors": list(SENSORS),
                "expected": {case: probs[index] for case, probs in predictions.items()}})
        assert len(segment_rows) == 8
        segments.append({"subject": subject, "target": raw_segment["target"], "windows": segment_rows})
    registry = {"schema": 1, "default_case": "full", "window_seconds": 60,
        "preprocessing": "Pinned Python SciPy/NeuroKit2 window preparation; full-sensor replay uses prepared features",
        "live_policy": "Use only compatible exposed sensors. Current standard API adapter supplies ACC; BPM is not BVP.",
        "models": models}
    (assets / "sensor_models.json").write_text(json.dumps(registry, indent=2))
    (assets / "sensor_replay.json").write_text(json.dumps({"schema": 1, "prepared_windows": True,
        "window_seconds": 60, "stride_seconds": 10, "segments": segments}))
    evidence = {"status": "python_onnx_validated_android_pending", "subject": subject,
        "evaluation_windows": len(windows), "replay_windows": 16, "models": models,
        "maximum_onnx_error": max(m["parity_max_error"] for m in models)}
    (output / "sensor_export.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def export_local_replay(assets: Path, data_root: Path) -> dict:
    """Prepare local WESAD replay using installed release models, without checkpoints."""
    from physio.processing.dataset import discover, load_recording
    from physio.processing.windows import WindowConfig, eligible_windows

    metadata = json.loads((assets / "model.json").read_text())
    subject = metadata["held_out_subject"]
    recording = load_recording(next(path for path in discover(data_root) if path.stem == subject))
    registry = json.loads((assets / "sensor_models.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = {}
    for model in registry["models"]:
        if model["held_out_subject"] != subject:
            raise ValueError("Replay subject must be held out by every model")
        filename = model["file"]
        if not re.fullmatch(r"sensor_models/[a-z_]+\.onnx", filename):
            raise ValueError("Invalid replay model path")
        data = (assets / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != model["model_sha256"]:
            raise ValueError("Replay model checksum mismatch")
        sessions[model["case"]] = ort.InferenceSession(
            data, options, providers=["CPUExecutionProvider"]
        )
    eligible = eligible_windows(recording.labels, WindowConfig())
    segments = []
    for target in (0, 1):
        starts = {start for start, label in eligible if label == target}
        start = next(
            (s for s in sorted(starts) if all(s + offset in starts for offset in range(0, 71, 10))),
            None,
        )
        if start is None:
            raise ValueError(f"No complete 130-second replay segment for class {target}")
        windows = []
        for offset in range(0, 71, 10):
            values = extract_features(slice_window(recording, start + offset, 60))
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("Nonfinite replay features")
            predictions = {}
            for model in registry["models"]:
                vector = np.array(
                    [[values[name] for name in model["feature_names"]]], dtype=np.float32
                )
                probability = float(sessions[model["case"]].run(None, {"features": vector})[0][0, 0])
                if not np.isfinite(probability) or not -1e-6 <= probability <= 1 + 1e-6:
                    raise ValueError("Invalid replay prediction")
                predictions[model["case"]] = float(np.clip(probability, 0, 1))
            windows.append({
                "start_seconds": start + offset, "target": target, "features": values,
                "available_sensors": list(SENSORS), "expected": predictions,
            })
        segments.append({"subject": subject, "target": target, "windows": windows})
    result = {
        "schema": 1, "prepared_windows": True, "window_seconds": 60,
        "stride_seconds": 10, "segments": segments,
    }
    (assets / "sensor_replay.json").write_text(json.dumps(result), encoding="utf-8")
    return {"subject": subject, "windows": sum(len(s["windows"]) for s in segments)}
