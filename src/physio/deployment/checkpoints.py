"""Load trusted local joblib checkpoints, including pre-cleanup model classes.

Uses the pinned joblib 1.4.2 loader to preserve NumPy storage and compression.
"""
from pathlib import Path
from typing import Any
from joblib.numpy_pickle import NumpyUnpickler, _read_fileobject


class CheckpointUnpickler(NumpyUnpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "physio.distillation" and name in {"Student", "FeatureMlp"}:
            from physio.training.tabular import FittedModel, FeatureMlp
            return {"Student": FittedModel, "FeatureMlp": FeatureMlp}[name]
        return super().find_class(module, name)


def load_checkpoint(path: str | Path) -> Any:
    """Read a trusted project checkpoint without changing its original bytes."""
    path = Path(path)
    with path.open("rb") as raw:
        with _read_fileobject(raw, str(path), None) as stream:
            if isinstance(stream, str):
                raise ValueError("Checkpoints older than joblib 0.10 are unsupported")
            return CheckpointUnpickler(str(path), stream, mmap_mode=None).load()
