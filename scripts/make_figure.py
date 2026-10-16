"""Figure 1: the benchmark's carbon label against ODIAC rebuilt over the same tiles.

One panel per city, one point per tile with both values positive. Circles mark tiles containing
a flagged point-source cell (K = 20, ODIAC2024, 2015). The dashed line is the median
label-per-tonne scaling of the tiles without one.

Output: figures/odiac_label.pdf
        figures/odiac_label.png

Run:
    python scripts/make_figure.py
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CITIES, CITY_LABEL, FIGURE_DIR, REPO_ROOT, configure_logging  # noqa: E402
from odiac import aggregate_to_tiles, load_odiac_city  # noqa: E402
from point_sources import detect_city, tiles_containing_cells  # noqa: E402

ORDINARY_COLOUR = "#0072B2"
SOURCE_COLOUR = "#D55E00"
STYLE = {
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def draw_panel(axis: plt.Axes, city: str) -> None:
    data = load_odiac_city(city)
    values, detection = detect_city(data)
    odiac_tiles = aggregate_to_tiles(values, data.grid, data.tile_bboxes)
    holds_source = tiles_containing_cells(detection, data.tile_bboxes)

    positive = (odiac_tiles > 0) & (data.carbon > 0)
    plain, source = positive & ~holds_source, positive & holds_source
    scaling = float(np.median(data.carbon[plain] / odiac_tiles[plain]))

    axis.scatter(odiac_tiles[plain], data.carbon[plain], s=4, c=ORDINARY_COLOUR, alpha=0.25,
                 linewidths=0, label="tile")
    axis.scatter(odiac_tiles[source], data.carbon[source], s=44, facecolors="none",
                 edgecolors=SOURCE_COLOUR, linewidths=1.4,
                 label="holds an injected point source")
    span = np.array([odiac_tiles[positive].min(), odiac_tiles[positive].max()])
    axis.plot(span, scaling * span, "--", color="0.35", linewidth=1.1, label="ordinary scaling")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(f"{CITY_LABEL[city]}  (n={int(positive.sum())})")
    axis.set_xlabel("ODIAC annual total over the tile (tonne C)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    configure_logging()
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update(STYLE)

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for axis, city in zip(axes, CITIES):
        draw_panel(axis, city)
    axes[0].set_ylabel("benchmark carbon label")
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.2)
    figure.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = FIGURE_DIR / f"odiac_label.{suffix}"
        figure.savefig(path, bbox_inches="tight")
        logging.info("wrote %s", path.relative_to(REPO_ROOT))
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
