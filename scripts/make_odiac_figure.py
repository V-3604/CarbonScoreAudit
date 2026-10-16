"""Figure 1: does the benchmark's carbon label carry ODIAC's injected point sources?

One panel per city. Each point is a tile: the ODIAC annual total rebuilt over that tile against
the benchmark's published carbon value. The dashed line is the scaling that tiles without a
detected point source follow. Tiles that hold one are drawn separately; in Beijing they sit on
the line, in Shanghai they fall one to two orders of magnitude below it, and New York does both.

Per-tile values are recomputed with the same functions the analysis uses, so the figure cannot
drift from the reported numbers. Panel n is smaller than the city tile count because the log axes
require both quantities to be strictly positive.

Input : data/external/odiac/crops/{version}/{city}_{YYYYMM}.npy
        artifacts/features/urbanmoe_{city}.npz
Output: figures/fig_urbanmoe_odiac.{pdf,png}

Run:
    python scripts/make_odiac_figure.py --year 2015
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plotting import CITY_LABEL, _save, _style  # noqa: E402
from urbanmoe_odiac_pointsources import DEFAULT_RATIO  # noqa: E402
from urbanmoe_odiac_pointsources import detect  # noqa: E402
from urbanmoe_odiac_vintage import (  # noqa: E402
    CITIES,
    PRIMARY_VERSION,
    aggregate_to_tiles,
    annual_grid,
    load_city,
)

ORDINARY = "#0072B2"
SOURCE = "#D55E00"


def panel(axis: plt.Axes, city: str, year: int, version: str, ratio_threshold: float) -> None:
    data = load_city(city)
    values = annual_grid(version, city, year, data.grid, sleep_s=0.0)
    odiac = aggregate_to_tiles(values, data.grid, data.tile_bboxes)
    detection = detect(values, data.grid, ratio_threshold)

    holds_source = np.zeros(len(odiac), dtype=bool)
    for lon, lat in zip(detection.lon, detection.lat):
        holds_source |= ((data.tile_bboxes[:, 0] <= lon) & (lon < data.tile_bboxes[:, 2])
                         & (data.tile_bboxes[:, 1] <= lat) & (lat < data.tile_bboxes[:, 3]))

    positive = (odiac > 0) & (data.carbon > 0)
    plain = positive & ~holds_source
    source = positive & holds_source
    baseline = float(np.median(data.carbon[plain] / odiac[plain]))

    axis.scatter(odiac[plain], data.carbon[plain], s=4, c=ORDINARY, alpha=0.25,
                 linewidths=0, label="tile")
    axis.scatter(odiac[source], data.carbon[source], s=44, facecolors="none",
                 edgecolors=SOURCE, linewidths=1.4,
                 label="holds an injected point source")

    grid = np.array([odiac[positive].min(), odiac[positive].max()])
    axis.plot(grid, baseline * grid, "--", color="0.35", linewidth=1.1,
              label="ordinary scaling")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(f"{CITY_LABEL[city]}  (n={int(positive.sum())})")
    axis.set_xlabel("ODIAC annual total over the tile (tonne C)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2015,
                        help="any covered year; the result is invariant across 2012-2022")
    parser.add_argument("--version", default=PRIMARY_VERSION)
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--stem", default="fig_urbanmoe_odiac")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for axis, city in zip(axes, CITIES):
        panel(axis, city, args.year, args.version, args.ratio)
    axes[0].set_ylabel("benchmark carbon label")
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.2)
    figure.tight_layout()
    _save(figure, args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
