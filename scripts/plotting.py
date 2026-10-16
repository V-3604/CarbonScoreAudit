"""Shared matplotlib styling and figure output for the paper figure."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"

CITY_LABEL = {"shanghai": "Shanghai", "beijing": "Beijing", "newyork": "New York"}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.dpi": 200,
    })


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        logging.info("wrote %s", path)
    plt.close(fig)
