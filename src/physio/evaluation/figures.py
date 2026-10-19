"""Consistent vector research plots; all inputs must come from saved results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "no-latex"])
sns.set_palette("colorblind")
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in [".svg", ".png"]:
        figure.savefig(path.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(figure)
