"""Train full-sensor or subset app MLPs with nested subject-independent evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.pycache_prefix = str(Path(__file__).resolve().parents[1] / "results/cache/python")

import joblib
import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import f1_score
import torch

from physio.training.tabular import fit_model
from physio.training.validation import grouped_splits
from physio.deployment.onnx_export import export_model
from physio.evaluation.metrics import metrics
from physio.evaluation.figures import plt, save_figure
from physio.processing.live_features import names_for
from physio.processing.sensor_contracts import SENSOR_CASES, feature_names, validate_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=["wesad", "live"], default="wesad")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sensor-set", choices=list(SENSOR_CASES), default="full")
    parser.add_argument("--subjects", nargs="+", help="Optional held-out folds; omit for complete LOSO")
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional final seeds; defaults to all configured seeds")
    args = parser.parse_args()
    if args.contract == "live":
        if args.sensor_set not in {"acc_bvp", "acc_temp", "acc_bvp_temp", "acc_eda", "acc_bvp_eda", "acc_eda_temp", "full"}:
            raise ValueError("Live models require ACC and one or more optional sensors; ACC baseline is preserved")
        args.features = args.features or Path("results/preprocessing/live_2025/features.csv")
        args.output = args.output or Path("results/training/live_sensor_subsets") / args.sensor_set
        names = names_for(SENSOR_CASES[args.sensor_set])
    else:
        args.features = args.features or Path("results/preprocessing/multisensor/features.csv"
                                             if args.sensor_set != "acc" else "results/preprocessing/watch/features.csv")
        args.output = args.output or Path("results/training/sensor_subsets") / args.sensor_set
        names = feature_names(SENSOR_CASES[args.sensor_set])
    config = json.loads(Path("configs/training.json").read_text())
    frame = pd.read_csv(args.features)
    if args.sensor_set != "full" and list(frame.columns[3:]) != names:
        frame = frame[["subject", "start_seconds", "target", *names]]
    validate_table(frame, names)
    subjects = sorted(frame.subject.unique())
    requested = args.subjects or subjects
    seeds = args.seeds or config["seeds"]
    if len(subjects) <= config["inner_subject_folds"] or not set(requested) <= set(subjects):
        raise ValueError("Need enough subjects for grouped tuning and valid held-out IDs")
    if not set(seeds) <= set(config["seeds"]):
        raise ValueError("Requested seeds must belong to configs/training.json")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "validated.json").exists():
        raise FileExistsError("Completed training is immutable; choose a new --output")
    feature_path = args.output / "features.csv"
    if feature_path.resolve() != args.features.resolve():
        if feature_path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(feature_path), frame)
        else:
            frame.to_csv(feature_path, index=False)
    manifest = {"config": config, "feature_names": names, "sensors": SENSOR_CASES[args.sensor_set], "subjects": subjects,
                "input_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
                "contract": args.contract, "supervised_only": True, "teacher_used": False}
    manifest_path = args.output / "training_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume inputs/configuration changed; choose a fresh --output")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    x, y, groups = frame[names].to_numpy(), frame.target.to_numpy(), frame.subject.to_numpy()
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    mlflow.set_tracking_uri(Path("results/mlflow").resolve().as_uri())
    mlflow.set_experiment("watch_" + args.contract + "_mlp_" + args.sensor_set)
    for subject in requested:
        train = np.flatnonzero(groups != subject)
        test = np.flatnonzero(groups == subject)
        folder = args.output / subject
        folder.mkdir(exist_ok=True)
        split_path = folder / "splits.json"
        if split_path.exists():
            audit = json.loads(split_path.read_text())
            selected = audit["selected"]
        else:
            splits = grouped_splits(groups[train], config["inner_subject_folds"])
            candidates = [{"family": "mlp",
                           "hidden": config["hidden"], "epochs": epochs,
                           "learning_rate": config["learning_rate"],
                           "weight_decay": config["weight_decay"]} for epochs in config["epochs"]]
            tuning = []
            for candidate in candidates:
                scores = []
                for a, b in splits:
                    a, b = train[a], train[b]
                    assert not set(groups[a]) & set(groups[b])
                    fitted = fit_model(x[a], y[a], candidate, config["tuning_seed"])
                    scores.append(f1_score(y[b], fitted.predict(x[b]) >= .5,
                                           average="macro", zero_division=0))
                tuning.append({"candidate": candidate, "inner_macro_f1": scores})
            selected = candidates[int(np.argmax([np.mean(row["inner_macro_f1"]) for row in tuning]))]
            audit = {"outer_train": sorted(set(groups[train])), "outer_test": subject,
                     "selected": selected, "tuning": tuning,
                     "inner": [{"train": sorted(set(groups[train[a]])),
                                "validation": sorted(set(groups[train[b]]))} for a, b in splits]}
            split_path.write_text(json.dumps(audit, indent=2))
        assert subject not in audit["outer_train"]
        for seed in seeds:
            score_path = folder / f"metrics_{seed}.json"
            if score_path.exists():
                required = [folder / f"mlp_supervised_{seed}.onnx",
                            folder / f"mlp_supervised_{seed}.joblib",
                            folder / f"predictions_{seed}.csv", folder / f"learning_{seed}.csv"]
                if not all(path.exists() for path in required):
                    raise FileNotFoundError(f"Incomplete saved model: {subject}, seed {seed}")
                continue
            fitted = fit_model(x[train], y[train], selected, seed)
            probability = fitted.predict(x[test])
            model_path = folder / f"mlp_supervised_{seed}.onnx"
            joblib.dump(fitted, folder / f"mlp_supervised_{seed}.joblib")
            export_model(fitted, model_path)
            session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
            exported = session.run(None, {"features": x[test].astype(np.float32)})[0].ravel()
            error = float(np.max(np.abs(probability - exported)))
            if error >= 5e-5 or not np.array_equal(probability >= .5, exported >= .5):
                raise ValueError(f"ONNX mismatch for {subject}, seed {seed}")
            result = {"model": "mlp_supervised", "subject": subject, "seed": seed,
                      **metrics(y[test], probability >= .5), "epochs": selected["epochs"],
                      "onnx_max_probability_error": error}
            pd.DataFrame({"subject": subject, "seed": seed, "row": test, "target": y[test],
                          "probability": probability}).to_csv(folder / f"predictions_{seed}.csv", index=False)
            pd.DataFrame(fitted.learning).to_csv(folder / f"learning_{seed}.csv", index=False)
            fig, axis = plt.subplots(figsize=(3.5, 2.5))
            history = pd.DataFrame(fitted.learning)
            axis.plot(history.epoch, history.hard_label_bce)
            axis.set(xlabel="Epoch", ylabel="Training BCE", title=f"Held out {subject}; seed {seed}")
            save_figure(fig, folder / f"learning_{seed}")
            with mlflow.start_run(run_name=f"{subject}_{seed}"):
                mlflow.log_params({"held_out": subject, "seed": seed, **selected})
                mlflow.log_metrics({key: float(value) for key, value in result.items()
                                   if isinstance(value, (int, float))})
                mlflow.log_artifact(str(model_path))
            score_path.write_text(json.dumps(result, indent=2))
            print(f"Trained {subject}, seed {seed}: macro-F1 {result['macro_f1']:.4f}", flush=True)
    scores = [json.loads(path.read_text()) for path in args.output.glob("S*/metrics_*.json")]
    folds = pd.DataFrame(scores)
    folds.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions = [pd.read_csv(path) for path in args.output.glob("S*/predictions_*.csv")]
    pd.concat(predictions, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    expected = {(subject, seed) for subject in subjects for seed in config["seeds"]}
    observed = set(zip(folds.subject, folds.seed))
    completed = observed == expected
    progress = {"status": "validated" if completed else "partial", "subjects": len(subjects),
                "completed_models": len(observed), "expected_models": len(expected),
                "seeds": config["seeds"], "contract": args.contract, "supervised_only": True, "teacher_used": False}
    (args.output / ("validated.json" if completed else "progress.json")).write_text(json.dumps(progress, indent=2))
    print(f"Saved {len(observed)}/{len(expected)} models to {args.output}")


if __name__ == "__main__":
    main()
