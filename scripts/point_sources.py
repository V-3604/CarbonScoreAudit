"""Find ODIAC's injected point sources in the raster and measure how each label carries them.

ODIAC places large point sources at inventory coordinates and spreads the remaining emissions by
nighttime lights, which is smooth at 1 km. An injected source therefore shows up as a single
cell far above its surroundings. A cell is flagged when its annual value is at least K times the
median of the ring of cells two to four cells away. The cell and its immediate neighbours are
left out of the ring so that a source does not raise its own background. The paper uses K = 20;
the report also lists the flagged cells for K = 5, 10, 50 and 100.

For each city, at ODIAC2024 and year 2015, the report gives:
  - the share of flagged cells within 5 km of a fossil plant of at least 50 MW in the WRI Global
    Power Plant Database, with renewable plants as a placebo;
  - for every tile containing a flagged cell, the label divided by the label expected at the
    city's ordinary label-per-tonne scaling, counted as carried (>= 0.5), partial or absent
    (< 0.1);
  - the city's label total against the total expected at that scaling;
  - the ordinary scaling itself, which differs between cities;
  - the same tile counts under ODIAC2022 and ODIAC2023.

Output: reports/point_sources.json

Run:
    python scripts/point_sources.py
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import mannwhitneyu

from common import (CITIES, PLANT_CSV, configure_logging, local_xy, nearest_km, tile_centres,
                    write_report)
from odiac import (CELL_DEG, ODIAC_VERSION, ODIAC_YEAR, VERSION_COVERAGE, CropGrid, OdiacCity,
                   aggregate_to_tiles, annual_grid, load_odiac_city)

RING_INNER, RING_OUTER = 2, 4
DETECTION_RATIO = 20.0
RATIO_SWEEP = (5.0, 10.0, 20.0, 50.0, 100.0)
CARRIED_AT_LEAST = 0.5
ABSENT_BELOW = 0.1

FOSSIL_FUELS = {"Coal", "Gas", "Oil", "Petcoke", "Cogeneration"}
RENEWABLE_FUELS = {"Solar", "Wind", "Hydro", "Nuclear", "Geothermal", "Biomass", "Wave and Tidal"}
MIN_FOSSIL_MW = 50.0
# Plants further than this from every tile centre lie outside the area the benchmark covers.
PLANT_MAX_KM_FROM_TILE = 2.0
CROSS_CHECK_KM = 5.0


@dataclass(frozen=True)
class Detection:
    """Flagged cells at one ratio threshold, as crop indices and cell-centre coordinates."""

    rows: np.ndarray
    cols: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    value: np.ndarray
    n_sources: int
    share_of_city_emission: float


def ring_background(values: np.ndarray) -> np.ndarray:
    """Median of the ring RING_INNER..RING_OUTER cells away (Chebyshev distance), per cell.

    Cells beyond the crop edge are NaN, so edge cells take their background from real cells only.
    """
    padded = np.pad(values, RING_OUTER, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (2 * RING_OUTER + 1, 2 * RING_OUTER + 1))
    offsets = np.arange(-RING_OUTER, RING_OUTER + 1)
    chebyshev = np.maximum(np.abs(offsets)[:, None], np.abs(offsets)[None, :])
    ring = (chebyshev >= RING_INNER) & (chebyshev <= RING_OUTER)
    flat = windows.reshape(values.shape[0], values.shape[1], -1)[..., ring.ravel()]
    with np.errstate(invalid="ignore"):
        return np.nanmedian(flat, axis=-1)


def detect(values: np.ndarray, grid: CropGrid, ratio_threshold: float) -> Detection:
    """Cells whose annual value is at least ratio_threshold times their ring background."""
    background = ring_background(values)
    usable = np.isfinite(background) & (background > 0) & (values > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(usable, values / background, 0.0)
    flagged = usable & (ratio >= ratio_threshold)
    rows, cols = np.nonzero(flagged)

    # Adjacent flagged cells (8-connectivity) count as one source.
    cells = set(zip(rows.tolist(), cols.tolist()))
    visited: set[tuple[int, int]] = set()
    n_sources = 0
    for cell in cells:
        if cell in visited:
            continue
        n_sources += 1
        stack = [cell]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            stack.extend((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                         if (r + dr, c + dc) in cells and (r + dr, c + dc) not in visited)

    total = float(values[values > 0].sum())
    return Detection(rows=rows, cols=cols,
                     lon=grid.west + (cols + 0.5) * CELL_DEG,
                     lat=grid.north - (rows + 0.5) * CELL_DEG,
                     value=values[rows, cols],
                     n_sources=n_sources,
                     share_of_city_emission=float(values[flagged].sum() / total))


def detect_city(data: OdiacCity, version: str = ODIAC_VERSION, year: int = ODIAC_YEAR,
                ratio_threshold: float = DETECTION_RATIO) -> tuple[np.ndarray, Detection]:
    """The annual ODIAC grid for one city and the cells flagged in it."""
    values = annual_grid(version, data.city, year, data.grid)
    return values, detect(values, data.grid, ratio_threshold)


def tiles_containing_cells(detection: Detection, tile_bboxes: np.ndarray) -> np.ndarray:
    """Tiles whose bounding box contains the centre of at least one flagged cell."""
    inside = np.zeros(len(tile_bboxes), dtype=bool)
    for lon, lat in zip(detection.lon, detection.lat):
        inside |= ((tile_bboxes[:, 0] <= lon) & (lon < tile_bboxes[:, 2])
                   & (tile_bboxes[:, 1] <= lat) & (lat < tile_bboxes[:, 3]))
    return inside


def point_mass_per_tile(values: np.ndarray, detection: Detection, data: OdiacCity) -> np.ndarray:
    """Flagged-cell emission inside each tile, area-weighted like the label rebuild.

    Positive for every tile that overlaps a flagged cell, which is the flag used in Table 2.
    """
    point_only = np.zeros_like(values)
    point_only[detection.rows, detection.cols] = values[detection.rows, detection.cols]
    return aggregate_to_tiles(point_only, data.grid, data.tile_bboxes)


def label_carrying(detection: Detection, odiac_tiles: np.ndarray, carbon: np.ndarray,
                   tile_bboxes: np.ndarray) -> dict[str, object]:
    """How much of the injected value the label holds on the tiles containing a flagged cell.

    The ordinary scaling is the median label-per-tonne ratio over tiles with ODIAC emission and
    no flagged cell. A source tile's shortfall is its label divided by its ODIAC value times
    that scaling, so a label that carries the injected value gives a shortfall near 1.
    """
    holds_source = tiles_containing_cells(detection, tile_bboxes)
    positive = odiac_tiles > 0
    ratio = carbon / np.where(positive, odiac_tiles, 1.0)
    ordinary = float(np.median(ratio[~holds_source & positive]))
    source = holds_source & positive
    shortfall = ratio[source] / ordinary
    p_lower = float(mannwhitneyu(ratio[source], ratio[~holds_source & positive],
                                 alternative="less").pvalue)

    expected_total = ordinary * float(odiac_tiles.sum())
    absent_mass = ordinary * float(odiac_tiles[source][shortfall < ABSENT_BELOW].sum())
    return {
        "n_source_tiles": int(source.sum()),
        "n_carried": int((shortfall >= CARRIED_AT_LEAST).sum()),
        "n_partial": int(((shortfall >= ABSENT_BELOW) & (shortfall < CARRIED_AT_LEAST)).sum()),
        "n_absent": int((shortfall < ABSENT_BELOW).sum()),
        "median_shortfall": float(np.median(shortfall)),
        "p_source_tiles_lower": p_lower,
        "label_per_tonne_c": ordinary,
        "label_total": float(carbon.sum()),
        "expected_label_total": expected_total,
        "deficit_frac": 1.0 - float(carbon.sum()) / expected_total,
        "deficit_frac_from_absent_sources": absent_mass / expected_total,
    }


def plants_near_tiles(plants: pd.DataFrame, tile_xy: np.ndarray, lat0: float) -> pd.DataFrame:
    plant_xy = local_xy(plants.longitude.to_numpy(), plants.latitude.to_numpy(), lat0)
    keep = np.zeros(len(plants), dtype=bool)
    for start in range(0, len(plants), 2000):
        keep[start:start + 2000] = (nearest_km(plant_xy[start:start + 2000], tile_xy)
                                    <= PLANT_MAX_KM_FROM_TILE)
    return plants[keep]


def plant_cross_check(detection: Detection, plants: pd.DataFrame, tile_xy: np.ndarray,
                      lat0: float) -> dict[str, object]:
    """Share of flagged cells within CROSS_CHECK_KM of a fossil plant, and of a renewable one."""
    nearby = plants_near_tiles(plants, tile_xy, lat0)
    cell_xy = local_xy(detection.lon, detection.lat, lat0)
    out: dict[str, object] = {}
    for label, fuels, min_mw in (("fossil", FOSSIL_FUELS, MIN_FOSSIL_MW),
                                 ("renewable", RENEWABLE_FUELS, 0.0)):
        subset = nearby[nearby.primary_fuel.isin(fuels) & (nearby.capacity_mw >= min_mw)]
        if subset.empty:
            out[label] = {"n_plants": 0, "min_capacity_mw": min_mw}
            continue
        plant_xy = local_xy(subset.longitude.to_numpy(), subset.latitude.to_numpy(), lat0)
        distance = nearest_km(cell_xy, plant_xy)
        out[label] = {
            "n_plants": int(len(subset)),
            "min_capacity_mw": min_mw,
            "frac_flagged_cells_near_a_plant": float((distance <= CROSS_CHECK_KM).mean()),
            "median_km_cell_to_plant": float(np.median(distance)),
        }
    return out


def analyse_city(data: OdiacCity, plants: pd.DataFrame) -> dict[str, object]:
    values, detection = detect_city(data)
    odiac_tiles = aggregate_to_tiles(values, data.grid, data.tile_bboxes)
    centres = tile_centres(data.tile_bboxes)
    lat0 = float(centres[:, 1].mean())
    tile_xy = local_xy(centres[:, 0], centres[:, 1], lat0)

    sweep = []
    for threshold in RATIO_SWEEP:
        swept = detect(values, data.grid, threshold)
        sweep.append({"ratio_threshold": threshold, "n_flagged_cells": int(len(swept.lon)),
                      "n_sources": swept.n_sources,
                      "share_of_city_emission": swept.share_of_city_emission})

    versions = {}
    for version, (first, last) in VERSION_COVERAGE.items():
        if first <= ODIAC_YEAR <= last:
            other_values, other_detection = detect_city(data, version=version)
            carrying = label_carrying(other_detection,
                                      aggregate_to_tiles(other_values, data.grid,
                                                         data.tile_bboxes),
                                      data.carbon, data.tile_bboxes)
            versions[version] = {k: carrying[k] for k in
                                 ("n_source_tiles", "n_carried", "n_partial", "n_absent")}

    carrying = label_carrying(detection, odiac_tiles, data.carbon, data.tile_bboxes)
    logging.info("%-9s %d flagged cells, %d source tiles: %d carried, %d partial, %d absent",
                 data.city, len(detection.lon), carrying["n_source_tiles"],
                 carrying["n_carried"], carrying["n_partial"], carrying["n_absent"])
    return {
        "n_tiles": int(len(data.carbon)),
        "n_tiles_both_positive": int(((odiac_tiles > 0) & (data.carbon > 0)).sum()),
        "n_flagged_cells": int(len(detection.lon)),
        "n_sources": detection.n_sources,
        "share_of_city_emission": detection.share_of_city_emission,
        "n_tiles_with_point_mass": int(
            (point_mass_per_tile(values, detection, data) > 0).sum()),
        "plant_cross_check": plant_cross_check(detection, plants, tile_xy, lat0),
        "label_carrying": carrying,
        "ratio_sweep": sweep,
        f"versions_at_{ODIAC_YEAR}": versions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plants-csv", type=Path, default=PLANT_CSV)
    args = parser.parse_args()
    configure_logging()

    if not args.plants_csv.exists():
        raise SystemExit(f"{args.plants_csv} not found. Download the WRI Global Power Plant "
                         f"Database v1.3.0 CSV into data/external/.")
    plants = pd.read_csv(args.plants_csv, low_memory=False)
    cities = {city: analyse_city(load_odiac_city(city), plants) for city in CITIES}

    scale = {city: cities[city]["label_carrying"]["label_per_tonne_c"] for city in CITIES}
    report = {
        "odiac_version": ODIAC_VERSION,
        "odiac_year": ODIAC_YEAR,
        "detection_ratio": DETECTION_RATIO,
        "ring_cells": [RING_INNER, RING_OUTER],
        "carried_at_least": CARRIED_AT_LEAST,
        "absent_below": ABSENT_BELOW,
        "newyork_label_scale_over": {"beijing": scale["newyork"] / scale["beijing"],
                                     "shanghai": scale["newyork"] / scale["shanghai"]},
        "cities": cities,
    }
    write_report("point_sources.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
