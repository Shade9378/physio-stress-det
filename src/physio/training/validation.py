"""Subject-grouped validation splits shared by training and experiments."""
import numpy as np
from sklearn.model_selection import GroupKFold

def grouped_splits(groups: np.ndarray, folds: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(np.unique(groups)) < folds:
        raise ValueError("Insufficient subjects for grouped validation")
    splits = list(GroupKFold(folds).split(np.zeros(len(groups)), groups=groups))
    for train, validation in splits:
        if set(groups[train]) & set(groups[validation]):
            raise ValueError("Subject leakage")
    return splits
