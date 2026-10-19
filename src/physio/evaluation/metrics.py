"""Subject-level summaries and reproducible plotting from saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {"accuracy": accuracy_score(target, prediction),
            "macro_f1": f1_score(target, prediction, average="macro", zero_division=0),
            "macro_precision": precision_score(target, prediction, average="macro", zero_division=0),
            "macro_recall": recall_score(target, prediction, average="macro", zero_division=0)}


def summarize(out: Path) -> pd.DataFrame:
    from physio.evaluation.figures import plt, save_figure, sns

    folds = pd.read_csv(out/"fold_metrics.csv")
    summary = folds.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        precision_mean=("macro_precision", "mean"), recall_mean=("macro_recall", "mean"),
        latency_ms_median=("latency_ms", "median"), model_bytes_mean=("model_bytes", "mean"))
    summary.to_csv(out/"summary.csv")
    fig, axis = plt.subplots(figsize=(3.5, 2.8))
    positions = np.arange(len(summary))
    axis.bar(positions, summary.macro_f1_mean, yerr=summary.macro_f1_std, capsize=2)
    axis.set_xticks(positions, summary.index, rotation=35, ha="right")
    axis.set(ylabel="Subject macro-F1 (mean ± SD)", ylim=(0, 1.1))
    save_figure(fig, out/"figures/model_comparison")
    fig, axis = plt.subplots(figsize=(7, 3))
    sns.stripplot(data=folds, x="model", y="macro_f1", ax=axis, jitter=False, size=4)
    axis.set(ylabel="Held-out subject macro-F1", ylim=(0, 1.1))
    axis.tick_params(axis="x", rotation=30)
    save_figure(fig, out/"figures/subject_performance")
    predictions = pd.read_csv(out/"predictions.csv")
    for model, frame in predictions.groupby("model"):
        matrix = confusion_matrix(frame.target, frame.prediction, labels=[0, 1])
        fig, axis = plt.subplots(figsize=(3.5, 2.8))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Baseline", "Stress"], yticklabels=["Baseline", "Stress"], ax=axis)
        axis.set(xlabel="Predicted", ylabel="Protocol condition", title=model)
        save_figure(fig, out/f"figures/confusion_{model}")
        (out/f"confusion_{model}.json").write_text(json.dumps(matrix.tolist()))
    return summary
