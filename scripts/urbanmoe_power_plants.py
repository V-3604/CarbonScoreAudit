"""Test whether the worst-predicted carbon tiles sit near real power plants.

ODIAC places large point sources at coordinates from an external plant inventory and spreads the
rest using nighttime lights. If the worst-predicted tiles sit on those point sources, the part of
the label no model reaches is a component injected from a database, at locations no satellite
image or POI count identifies. That is a statement about the target, not about any architecture.

Reference inventory is the WRI Global Power Plant Database (v1.3.0, CC-BY 4.0), with coordinates
in EPSG:4326.

Placebo control: renewable plants (solar, wind, hydro, nuclear, geothermal) emit no CO2 but are
sited under similar land-availability pressure. A real effect appears for fossil plants and not
for renewables; if it appears for both, the finding is about where plants get built.

Protocol note: residuals here come from a 5-fold cross-validated probe, not the 60/20/20
ten-seed protocol used for every score reported in the paper's tables. The two are never mixed.

Input : artifacts/features/urbanmoe_{city}.npz
        data/external/global_power_plant_database.csv
Output: artifacts/reports/urbanmoe_power_plants.json

Run:
    python scripts/urbanmoe_power_plants.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import load_centroids  # noqa: E402
from urbanmoe_probe import probe_target  # noqa: E402

FEATURE_DIR = REPO_ROOT / "artifacts/features"
REPORT_DIR = REPO_ROOT / "artifacts/reports"
PLANT_CSV = REPO_ROOT / "data/external/global_power_plant_database.csv"

CITIES = ("shanghai", "beijing", "newyork")
SEED = 42
N_FOLDS = 5
CARBON = 0
N_WORST = 25
# Tiles are 2 km squares, so this keeps plants sitting within roughly one tile of the sampled grid
# and drops the rest. Not arbitrary: the plant-to-nearest-tile distances break naturally, New York
# runs 0.4-1.1 km then jumps to 7.9-9.5 km with nothing between, Beijing 0.5-1.1 km then 4.1 km.
# The dropped ones sit over map the benchmark never sampled, where there is no imagery, no POI and
# no carbon label, so there is no residual to draw beneath them and they read as floating markers.
# Conclusion is invariant to the choice: at a 10 km radius Beijing is p=1.8e-14 and New York
# p=2.7e-09, at 2 km they are p=1.6e-14 and p=3.0e-09. Same rule for the stats and the figure.
PLANT_MAX_KM_FROM_TILE = 2.0
# New York lists 73 fossil plants near tiles but their MEDIAN capacity is 13 MW and the smallest is
# 1 MW -- peakers and standby sets that contribute almost nothing to annual CO2, while ODIAC's
# point-source term is emissions-weighted. Beijing's 12 plants are all >=150 MW, so this threshold
# changes Beijing not at all and only de-clutters New York. CAPACITY_SWEEP_MW records that the
# conclusion does not depend on the choice; do not report the 50 MW row without that sweep.
MIN_FOSSIL_MW = 50.0
CAPACITY_SWEEP_MW = (0.0, 50.0, 100.0, 300.0)

FOSSIL_FUELS = {"Coal", "Gas", "Oil", "Petcoke", "Cogeneration"}
RENEWABLE_FUELS = {"Solar", "Wind", "Hydro", "Nuclear", "Geothermal", "Biomass", "Wave and Tidal"}


def local_xy(lon: np.ndarray, lat: np.ndarray, lat0: float) -> np.ndarray:
    """Equirectangular km grid about lat0. Distances in km, never degrees."""
    return np.column_stack([lon * 111.320 * np.cos(np.radians(lat0)), lat * 110.540])


def nearest_km(tile_xy: np.ndarray, plant_xy: np.ndarray) -> np.ndarray:
    if len(plant_xy) == 0:
        return np.full(len(tile_xy), np.inf)
    d = np.sqrt(((tile_xy[:, None, :] - plant_xy[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1)


def plants_near_tiles(plants: pd.DataFrame, tile_xy: np.ndarray, lat0: float) -> pd.DataFrame:
    """Plants within PLANT_MAX_KM_FROM_TILE of any sampled tile centroid.

    The tile grids are irregular blobs, not filled rectangles, so a plain bounding box admits
    plants sitting over water or over unsampled countryside. Those can never be the nearest plant
    to a tile that matters, and on the map they read as unexplained floating markers.
    """
    plant_xy = local_xy(plants.longitude.to_numpy(), plants.latitude.to_numpy(), lat0)
    keep = np.empty(len(plants), dtype=bool)
    for i in range(0, len(plants), 2000):
        chunk = plant_xy[i:i + 2000]
        d = np.sqrt(((chunk[:, None, :] - tile_xy[None, :, :]) ** 2).sum(axis=2))
        keep[i:i + 2000] = d.min(axis=1) <= PLANT_MAX_KM_FROM_TILE
    return plants[keep]


def carbon_residuals(city: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centroids, out-of-fold residual, true carbon). Residual = true - predicted.

    5-fold CV, not the benchmark's 60/20/20 x 10 seeds, for one reason: every tile needs exactly
    one prediction from a model that never saw it. Under 10 random 20% test folds, 0.8^10 = 11% of
    tiles never land in any test set, which left real holes in the map. Scored benchmark numbers
    everywhere else still use the published 60/20/20 protocol; this is a diagnostic.
    """
    bundle = np.load(FEATURE_DIR / f"urbanmoe_{city}.npz", allow_pickle=True)
    image, poi, y = bundle["image"], bundle["poi"], bundle["y"]
    carbon = y[:, CARBON]
    folds = [(train, test) for train, test in
             KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(carbon)]
    stats = probe_target(np.hstack([image, poi]), carbon,
                         block_sizes=(image.shape[1], poi.shape[1]),
                         target_transform="log1p", splits=folds, return_predictions=True)
    residual = np.full(len(carbon), np.nan)
    for test, pred in stats["fold_predictions"]:
        residual[test] = carbon[test] - pred
    if not np.isfinite(residual).all():
        raise RuntimeError(f"{city}: {np.isnan(residual).sum()} tiles have no out-of-fold "
                           f"prediction; K-fold should cover every row exactly once")
    return load_centroids(city, bundle["files"]), residual, carbon


def analyse_city(city: str, plants: pd.DataFrame) -> dict[str, object]:
    centroids, residual, carbon = carbon_residuals(city)
    lat0 = float(centroids[:, 1].mean())
    tile_xy = local_xy(centroids[:, 0], centroids[:, 1], lat0)

    in_region = plants_near_tiles(plants, tile_xy, lat0)

    worst = np.argsort(-np.abs(residual))[:N_WORST]
    others = np.setdiff1d(np.arange(len(residual)), worst)

    groups: dict[str, dict[str, object]] = {}
    distances: dict[str, np.ndarray] = {}
    # The renewable placebo is deliberately NOT capacity-filtered: renewables here are all small,
    # so a threshold would empty the control. Keeping every one of them is the conservative choice
    # (more plants = more opportunity for a spurious effect to appear).
    for label, fuels, floor in (("fossil", FOSSIL_FUELS, MIN_FOSSIL_MW),
                                ("renewable", RENEWABLE_FUELS, 0.0)):
        subset = in_region[in_region.primary_fuel.isin(fuels)
                           & (in_region.capacity_mw >= floor)]
        plant_xy = local_xy(subset.longitude.to_numpy(), subset.latitude.to_numpy(), lat0)
        distances[label] = nearest_km(tile_xy, plant_xy)
        groups[label] = {
            "n_plants": int(len(subset)),
            "min_capacity_mw_applied": float(floor),
            "total_capacity_mw": float(subset.capacity_mw.sum()),
        }

    sweep = []
    for floor in CAPACITY_SWEEP_MW:
        subset = in_region[in_region.primary_fuel.isin(FOSSIL_FUELS)
                           & (in_region.capacity_mw >= floor)]
        if len(subset) < 3:
            continue
        d = nearest_km(tile_xy, local_xy(subset.longitude.to_numpy(),
                                         subset.latitude.to_numpy(), lat0))
        sweep.append({
            "min_capacity_mw": floor,
            "n_plants": int(len(subset)),
            "median_km_worst_tiles": float(np.median(d[worst])),
            "median_km_other_tiles": float(np.median(d[others])),
            "p_worst_closer": float(mannwhitneyu(d[worst], d[others], alternative="less")[1]),
            "frac_worst_within_10km": float((d[worst] <= 10).mean()),
            "frac_other_within_10km": float((d[others] <= 10).mean()),
        })

    for label in ("fossil", "renewable"):
        d = distances[label]
        if not np.isfinite(d).any():
            groups[label].update({"testable": False})
            continue
        worst_d, other_d = d[worst], d[others]
        stat, p = mannwhitneyu(worst_d, other_d, alternative="less")
        groups[label].update({
            "testable": True,
            "median_km_worst_tiles": float(np.median(worst_d)),
            "median_km_other_tiles": float(np.median(other_d)),
            "mannwhitney_u": float(stat),
            "p_worst_closer": float(p),
            "significant_at_0_05": bool(p < 0.05),
            "frac_worst_within_10km": float((worst_d <= 10).mean()),
            "frac_other_within_10km": float((other_d <= 10).mean()),
        })

    return {
        "city": city,
        "n_tiles": int(len(carbon)),
        "groups": groups,
        "fossil_capacity_sweep": sweep,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plants-csv", type=Path, default=PLANT_CSV)
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "urbanmoe_power_plants.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if not args.plants_csv.exists():
        raise SystemExit(f"{args.plants_csv} not found. Download the WRI Global Power Plant "
                         f"Database CSV (v1.3.0) into data/external/ first.")

    plants = pd.read_csv(args.plants_csv, low_memory=False)
    cities = {}
    for city in CITIES:
        logging.info("analysing %s", city)
        cities[city] = analyse_city(city, plants)
        for label in ("fossil", "renewable"):
            g = cities[city]["groups"][label]
            if g.get("testable"):
                logging.info("  %-9s n=%3d  worst %.1f km vs others %.1f km  p=%.4g",
                             label, g["n_plants"], g["median_km_worst_tiles"],
                             g["median_km_other_tiles"], g["p_worst_closer"])

    payload = {
        "question": "do the worst carbon-residual tiles sit closer to real power plants than "
                    "other tiles, and does the effect survive a renewable-only placebo?",
        "why_this_is_not_circular": "ODIAC places point sources at plant coordinates from an "
                                    "external inventory and downscales the rest by nighttime "
                                    "lights. Showing the failures sit on the point sources shows "
                                    "the unpredictable part of the LABEL is injected from a "
                                    "database, not that power plants emit CO2.",
        "source": "WRI Global Power Plant Database v1.3.0, CC-BY 4.0",
        "protocol": {"seed": SEED, "n_folds": N_FOLDS, "n_worst": N_WORST,
                     "residual": "out-of-fold, 5-fold CV so every tile is predicted exactly once, image+POI ridge, log1p carbon target",
                     "test": "Mann-Whitney U, one-sided (worst tiles closer)",
                     "plant_inclusion": f"within {PLANT_MAX_KM_FROM_TILE} km of a sampled tile"},
        "cities": cities,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logging.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
