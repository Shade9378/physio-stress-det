"""Supervised tabular MLP and linear training with training-only normalization."""
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import torch
from torch import nn

class FeatureMlp(nn.Module):
    def __init__(self, hidden: int = 16, input_features: int = 6) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


@dataclass
class FittedModel:
    scaler: StandardScaler
    model: Any
    candidate: dict
    seed: int
    learning: list[dict]

    def predict(self, x: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(x)
        if self.candidate["family"] == "linear":
            return expit(scaled @ self.model[:-1] + self.model[-1])
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(scaled, dtype=torch.float32))).numpy()


def linear_objective(parameters: np.ndarray, x: np.ndarray, y: np.ndarray,
                     weights: np.ndarray, regularization_c: float) -> tuple[float, np.ndarray]:
    """Balanced supervised BCE with the original sklearn-compatible L2 penalty."""
    if regularization_c <= 0:
        raise ValueError("Regularization C must be positive")
    logits = x @ parameters[:-1] + parameters[-1]
    total = weights.sum()
    penalty = np.sum(parameters[:-1] ** 2) / (2 * regularization_c * total)
    loss = np.sum(weights * (np.logaddexp(0, logits) - y * logits)) / total + penalty
    residual = weights * (expit(logits) - y) / total
    gradient = np.r_[x.T @ residual + parameters[:-1] / (regularization_c * total), residual.sum()]
    return float(loss), gradient


def fit_model(x: np.ndarray, y: np.ndarray, candidate: dict, seed: int) -> FittedModel:
    if candidate.get("alpha", 0.) != 0.:
        raise ValueError("Only supervised loss is supported")
    if candidate["family"] not in {"linear", "mlp"}:
        raise ValueError("Expected linear or mlp model family")
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    weights = compute_sample_weight("balanced", y)
    if candidate["family"] == "linear":
        fitted = minimize(linear_objective, np.zeros(x.shape[1] + 1),
                          args=(scaled, y, weights, candidate["c"]),
                          jac=True, method="L-BFGS-B",
                          options={"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-12})
        if not fitted.success:
            raise RuntimeError(f"Model optimization failed: {fitted.message}")
        return FittedModel(scaler, fitted.x, candidate, seed, [])
    torch.manual_seed(seed)
    model = FeatureMlp(candidate["hidden"], input_features=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"],
                                 weight_decay=candidate["weight_decay"])
    features = torch.tensor(scaled, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32)
    sample_weights = torch.tensor(weights, dtype=torch.float32)
    history = []
    for epoch in range(candidate["epochs"]):
        optimizer.zero_grad()
        logits = model(features)
        hard = (nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none") * sample_weights).sum() / sample_weights.sum()
        loss = hard
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "hard_label_bce": float(hard.detach()),
                        "objective": float(loss.detach())})
    model.eval()
    return FittedModel(scaler, model, candidate, seed, history)
