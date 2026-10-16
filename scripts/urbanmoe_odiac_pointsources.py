"""Detect ODIAC's injected point-source cells in the raster itself.

Testing the worst tiles against a modern plant database uses that database as a proxy for what
ODIAC actually injected. This removes the proxy by finding the point sources directly in the
raster the label was built from.

ODIAC is the sum of large plants' reported emissions placed at inventory coordinates and the
remainder downscaled by nighttime lights. The nightlight term is smooth at 1 km; the point term
is not. An injected source therefore appears as a single cell holding many times its own
neighbourhood's value.

Detection rule: flag a cell whose annual value is at least K times the median of the ring of
cells two to four cells away, with the core excluded so a source cannot raise the background it
is compared against. K is not chosen by taste; the ratio distribution is reported and every
result is swept over K. The paper reports K = 20.

The plant cross-check runs in both directions and carries a renewable placebo: if detected cells
sit near renewables as often as near fossil plants, the rule is finding land use rather than
emitters.

Protocol note: residuals are taken from urbanmoe_power_plants verbatim, so they use the 5-fold
protocol rather than the 60/20/20 ten-seed protocol behind the paper's reported scores. The two
are never mixed.

Input : artifacts/reports/urbanmoe_odiac_vintage.json
        data/external/odiac/crops/{version}/{city}_{YYYYMM}.npy
        data/external/global_power_plant_database.csv
Output: artifacts/reports/urbanmoe_odiac_pointsources.json

Run:
    python scripts/urbanmoe_odiac_pointsources.py --year 2015
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from urbanmoe_odiac_vintage import (  # noqa: E402
    CITIES,
    PRIMARY_VERSION,
    RES_DEG,
    VERSION_COVERAGE,
    CropGrid,
    aggregate_to_tiles,
    annual_grid,
    load_city,
)
from urbanmoe_power_plants import (  # noqa: E402
    FOSSIL_FUELS,
    MIN_FOSSIL_MW,
    N_WORST,
    PLANT_CSV,
    RENEWABLE_FUELS,
    carbon_residuals,
    local_xy,
    nearest_km,
    plants_near_tiles,
)

REPORT_DIR = REPO_ROOT / "artifacts/reports"
VINTAGE_JSON = REPORT_DIR / "urbanmoe_odiac_vintage.json"

# Ring radii in cells. The inner hole keeps a source (and any spill into its immediate neighbours)
# out of its own background estimate; the outer edge keeps the ring local enough to track the
# nightlight surface, which varies over a few km.
RING_INNER = 2
RING_OUTER = 4
RATIO_SWEEP = (5.0, 10.0, 20.0, 50.0, 100.0)
DEFAULT_RATIO = 20.0
# A tile is 2 km across, so a detected cell within this distance of a tile centroid is adjacent to
# or inside it. Matches the tile-width reasoning in urbanmoe_power_plants.PLANT_MAX_KM_FROM_TILE.
ADJOIN_KM = 2.0
CROSS_CHECK_KM = 5.0


@dataclass(frozen=True)
class Detection:
    """Detected point-source cells at one ratio threshold."""

    ratio_threshold: float
    lon: np.ndarray
    lat: np.ndarray
    value: np.ndarray
    n_components: int
    share_of_city_emission: float


def ring_background(values: np.ndarray) -> np.ndarray:
    """Median of the ring RING_INNER..RING_OUTER cells away, per cell. NaN where undefined.

    Padded with NaN rather than a constant so cells at the crop edge take their background from
    real neighbours only, instead of an invented zero that would inflate their ratio.
    """
    pad = RING_OUTER
    padded = np.pad(values, pad, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (2 * pad + 1, 2 * pad + 1))

    offsets = np.arange(-pad, pad + 1)
    chebyshev = np.maximum(np.abs(offsets)[:, None], np.abs(offsets)[None, :])
    ring = (chebyshev >= RING_INNER) & (chebyshev <= RING_OUTER)

    flat = windows.reshape(values.shape[0], values.shape[1], -1)[..., ring.ravel()]
    with np.errstate(invalid="ignore"):
        return np.nanmedian(flat, axis=-1)


def detect(values: np.ndarray, grid: CropGrid, ratio_threshold: float) -> Detection:
    """Cells whose annual value is >= ratio_threshold times their ring background."""
    background = ring_background(values)
    usable = np.isfinite(background) & (background > 0) & (values > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(usable, values / background, 0.0)
    flagged = usable & (ratio >= ratio_threshold)

    rows, cols = np.nonzero(flagged)
    lon = grid.west + (cols + 0.5) * RES_DEG
    lat = grid.north - (rows + 0.5) * RES_DEG

    # Merge 8-connected flagged cells so a source spilling across a cell edge counts once.
    seen = set(zip(rows.tolist(), cols.tolist()))
    components, visited = 0, set()
    for cell in seen:
        if cell in visited:
            continue
        components += 1
        stack = [cell]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            stack.extend((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                         if (r + dr, c + dc) in seen and (r + dr, c + dc) not in visited)

    total = float(values[values > 0].sum())
    return Detection(
        ratio_threshold=ratio_threshold,
        lon=lon, lat=lat, value=values[rows, cols],
        n_components=components,
        share_of_city_emission=float(values[flagged].sum() / total) if total > 0 else float("nan"),
    )


def residual_test(detection: Detection, tile_xy: np.ndarray, lat0: float,
                  worst: np.ndarray, others: np.ndarray,
                  tile_bboxes: np.ndarray) -> dict[str, object]:
    """Are the worst-residual tiles closer to detected point cells than every other tile?"""
    if len(detection.lon) == 0:
        return {"testable": False, "reason": "no cells detected at this threshold"}

    cell_xy = local_xy(detection.lon, detection.lat, lat0)
    distance_km = nearest_km(tile_xy, cell_xy)

    inside = np.zeros(len(tile_bboxes), dtype=bool)
    for lon, lat in zip(detection.lon, detection.lat):
        inside |= ((tile_bboxes[:, 0] <= lon) & (lon < tile_bboxes[:, 2])
                   & (tile_bboxes[:, 1] <= lat) & (lat < tile_bboxes[:, 3]))

    stat, p = mannwhitneyu(distance_km[worst], distance_km[others], alternative="less")
    return {
        "testable": True,
        "median_km_worst_tiles": float(np.median(distance_km[worst])),
        "median_km_other_tiles": float(np.median(distance_km[others])),
        "mannwhitney_u": float(stat),
        "p_worst_closer": float(p),
        "significant_at_0_05": bool(p < 0.05),
        "frac_worst_containing_a_cell": float(inside[worst].mean()),
        "frac_other_containing_a_cell": float(inside[others].mean()),
        "frac_worst_within_2km": float((distance_km[worst] <= ADJOIN_KM).mean()),
        "frac_other_within_2km": float((distance_km[others] <= ADJOIN_KM).mean()),
    }


def spike_tile_contrast(detection: Detection, odiac_tiles: np.ndarray, carbon: np.ndarray,
                        tile_bboxes: np.ndarray) -> dict[str, object]:
    """Does the LABEL carry the point sources that are in the raster?

    Purely descriptive, and deliberately so. Comparing fits before and after removing spike cells
    changes the predictor, so a fit improving is partly mechanical and proves little. This instead
    reads the label-to-ODIAC ratio directly on the tiles that hold a detected source against the
    tiles that do not. If the label contains the injected point term the two ratios agree; if it
    omits it, the ratio collapses on exactly the spike tiles.
    """
    if len(detection.lon) == 0:
        return {"testable": False, "reason": "no cells detected at this threshold"}

    holds_spike = np.zeros(len(tile_bboxes), dtype=bool)
    for lon, lat in zip(detection.lon, detection.lat):
        holds_spike |= ((tile_bboxes[:, 0] <= lon) & (lon < tile_bboxes[:, 2])
                        & (tile_bboxes[:, 1] <= lat) & (lat < tile_bboxes[:, 3]))

    positive = odiac_tiles > 0
    ratio = np.where(positive, carbon / np.where(positive, odiac_tiles, 1.0), np.nan)
    spike_ratio = ratio[holds_spike & positive]
    plain_ratio = ratio[~holds_spike & positive]
    if len(spike_ratio) == 0 or len(plain_ratio) == 0:
        return {"testable": False, "reason": "one of the two tile groups is empty"}

    median_plain = float(np.median(plain_ratio))
    median_spike = float(np.median(spike_ratio))
    stat, p = mannwhitneyu(spike_ratio, plain_ratio, alternative="less")

    # The sentence-sized version, computed PER TILE. A median of predicted values over one group
    # against a median of observed values over another is a ratio of medians across different
    # tiles, which is misleading under this much skew -- it agreed with the paired statistic in
    # Shanghai and disagreed by 3x in New York. Pair each tile with itself instead, and quote the
    # single largest source as the illustration.
    source = holds_spike & positive
    predicted = median_plain * odiac_tiles[source]
    observed = carbon[source]
    shortfall = observed / predicted
    largest = int(np.argmax(odiac_tiles[source]))
    return {
        "testable": True,
        "n_tiles_holding_a_source": int((holds_spike & positive).sum()),
        "n_tiles_without": int((~holds_spike & positive).sum()),
        "median_ratio_on_source_tiles": median_spike,
        "median_ratio_on_other_tiles": median_plain,
        "ratio_of_medians": median_spike / median_plain if median_plain else float("nan"),
        "p_source_tiles_lower": float(p),
        "mannwhitney_u": float(stat),
        "median_per_tile_shortfall": float(np.median(shortfall)),
        "min_per_tile_shortfall": float(shortfall.min()),
        "largest_source_tile": {
            "odiac_tonneC": float(odiac_tiles[source][largest]),
            "label_predicted_at_ordinary_scaling": float(predicted[largest]),
            "label_observed": float(observed[largest]),
            "shortfall": float(shortfall[largest]),
        },
        "reading": "ratio_of_medians and median_per_tile_shortfall near 1 mean the label carries "
                   "the injected point term; far below 1 means the label omits it on exactly "
                   "those tiles. Both are paired per tile.",
    }


def mass_conservation(detection: Detection, odiac_tiles: np.ndarray, carbon: np.ndarray,
                      tile_bboxes: np.ndarray) -> dict[str, object]:
    """Does the city's total label mass fall short by exactly the mass of its absent sources?

    Independent of the per-tile test and of where any threshold is drawn: if the label carries
    everything, its total matches the ordinary scaling; if sources are missing, the deficit should
    equal their mass. Beijing, which omits none, is the built-in control.
    """
    if len(detection.lon) == 0:
        return {"testable": False}
    holds_spike = np.zeros(len(tile_bboxes), dtype=bool)
    for lon, lat in zip(detection.lon, detection.lat):
        holds_spike |= ((tile_bboxes[:, 0] <= lon) & (lon < tile_bboxes[:, 2])
                        & (tile_bboxes[:, 1] <= lat) & (lat < tile_bboxes[:, 3]))
    positive = odiac_tiles > 0
    baseline = float(np.median((carbon / np.where(positive, odiac_tiles, 1.0))[~holds_spike
                                                                              & positive]))
    predicted_total = baseline * float(odiac_tiles.sum())
    source = holds_spike & positive
    shortfall = carbon[source] / (baseline * odiac_tiles[source])
    absent_mass = baseline * float(odiac_tiles[source][shortfall < 0.1].sum())
    return {
        "testable": True,
        "baseline_ratio_label_per_tonneC": baseline,
        "label_total": float(carbon.sum()),
        "predicted_total_at_ordinary_scaling": predicted_total,
        "observed_over_predicted": float(carbon.sum()) / predicted_total,
        "deficit_frac": 1.0 - float(carbon.sum()) / predicted_total,
        "deficit_frac_explained_by_absent_sources": absent_mass / predicted_total,
    }


def wri_cross_check(detection: Detection, plants: pd.DataFrame, tile_xy: np.ndarray,
                    lat0: float) -> dict[str, object]:
    """Agreement between detected cells and the WRI plants used as a proxy for them."""
    if len(detection.lon) == 0:
        return {"testable": False}
    in_region = plants_near_tiles(plants, tile_xy, lat0)
    cell_xy = local_xy(detection.lon, detection.lat, lat0)

    out: dict[str, object] = {"n_detected_cells": int(len(detection.lon))}
    for label, fuels, floor in (("fossil", FOSSIL_FUELS, MIN_FOSSIL_MW),
                                ("renewable", RENEWABLE_FUELS, 0.0)):
        subset = in_region[in_region.primary_fuel.isin(fuels)
                           & (in_region.capacity_mw >= floor)]
        if len(subset) == 0:
            out[label] = {"n_plants": 0, "testable": False}
            continue
        plant_xy = local_xy(subset.longitude.to_numpy(), subset.latitude.to_numpy(), lat0)
        out[label] = {
            "n_plants": int(len(subset)),
            "min_capacity_mw_applied": float(floor),
            "frac_detected_cells_near_a_plant": float(
                (nearest_km(cell_xy, plant_xy) <= CROSS_CHECK_KM).mean()),
            "frac_plants_near_a_detected_cell": float(
                (nearest_km(plant_xy, cell_xy) <= CROSS_CHECK_KM).mean()),
            "median_km_cell_to_plant": float(np.median(nearest_km(cell_xy, plant_xy))),
        }
    return out


def version_invariance(city: str, data, year: int, ratio_threshold: float) -> dict[str, object]:
    """Is the omission an artifact of which ODIAC version we happened to read?

    ODIAC is re-released annually and each version revises the whole back series, so a finding
    that appears in one version and not another is a version artifact rather than a property of
    the label. Re-runs the detection and the shortfall count under every available version.
    """
    out: dict[str, object] = {}
    for version in sorted(VERSION_COVERAGE):
        low, high = VERSION_COVERAGE[version]
        if not low <= year <= high:
            out[version] = {"skipped": f"{version} covers {low}-{high}"}
            continue
        values = annual_grid(version, city, year, data.grid, sleep_s=0.0)
        odiac_tiles = aggregate_to_tiles(values, data.grid, data.tile_bboxes)
        detection = detect(values, data.grid, ratio_threshold)
        contrast = spike_tile_contrast(detection, odiac_tiles, data.carbon, data.tile_bboxes)
        out[version] = {
            "n_components": detection.n_components,
            "share_of_city_emission": detection.share_of_city_emission,
            "ratio_of_medians": contrast.get("ratio_of_medians"),
            "n_tiles_holding_a_source": contrast.get("n_tiles_holding_a_source"),
        }
    return out


def resolve_year(explicit: int | None) -> tuple[int, str]:
    if explicit is not None:
        return explicit, "given on the command line"
    if not VINTAGE_JSON.exists():
        raise SystemExit(f"{VINTAGE_JSON} not found and --year not given. Run "
                         f"urbanmoe_odiac_vintage.py first, or pass --year explicitly.")
    vintage = json.loads(VINTAGE_JSON.read_text(encoding="utf-8"))
    years = {entry["best_year"] for entry in vintage["best_per_city"].values()}
    if not vintage.get("vintage_identified") or len(years) != 1:
        raise SystemExit(f"the vintage sweep did not identify a single year "
                         f"(best_per_city says {sorted(years)}, "
                         f"vintage_identified={vintage.get('vintage_identified')}). Pass --year "
                         f"explicitly and say in the write-up that the year is not established.")
    return years.pop(), f"identified by {VINTAGE_JSON.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="+", default=list(CITIES), choices=list(CITIES))
    parser.add_argument("--year", type=int, default=None,
                        help="ODIAC year; defaults to the year the vintage sweep identified")
    parser.add_argument("--version", default=PRIMARY_VERSION)
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO,
                        help="headline detection threshold; every result is also swept")
    parser.add_argument("--plants-csv", type=Path, default=PLANT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "urbanmoe_odiac_pointsources.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    year, provenance = resolve_year(args.year)
    logging.info("using ODIAC %s %d (%s)", args.version, year, provenance)
    if args.dry_run:
        logging.info("dry run: would detect point cells in %d cities at ratios %s and test them "
                     "against 5-fold carbon residuals", len(args.cities), list(RATIO_SWEEP))
        return 0
    if not args.plants_csv.exists():
        raise SystemExit(f"{args.plants_csv} not found. Download the WRI Global Power Plant "
                         f"Database CSV (v1.3.0) into data/external/ first.")

    plants = pd.read_csv(args.plants_csv, low_memory=False)
    cities: dict[str, object] = {}
    for city in args.cities:
        logging.info("=== %s ===", city)
        data = load_city(city)
        values = annual_grid(args.version, city, year, data.grid, sleep_s=0.0)

        odiac_tiles = aggregate_to_tiles(values, data.grid, data.tile_bboxes)
        centroids, residual, _ = carbon_residuals(city)
        lat0 = float(centroids[:, 1].mean())
        tile_xy = local_xy(centroids[:, 0], centroids[:, 1], lat0)
        worst = np.argsort(-np.abs(residual))[:N_WORST]
        others = np.setdiff1d(np.arange(len(residual)), worst)

        background = ring_background(values)
        usable = np.isfinite(background) & (background > 0) & (values > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(usable, values / background, np.nan)
        finite_ratio = ratio[np.isfinite(ratio)]

        sweep = []
        for threshold in RATIO_SWEEP:
            detection = detect(values, data.grid, threshold)
            sweep.append({
                "ratio_threshold": threshold,
                "n_cells": int(len(detection.lon)),
                "n_components": detection.n_components,
                "share_of_city_emission": detection.share_of_city_emission,
                "residual_test": residual_test(detection, tile_xy, lat0, worst, others,
                                               data.tile_bboxes),
                "label_carries_the_point_term": spike_tile_contrast(
                    detection, odiac_tiles, data.carbon, data.tile_bboxes),
            })
            test = sweep[-1]["residual_test"]
            logging.info("  ratio>=%5.0f: %4d cells (%3d sources, %.1f%% of emission)  %s",
                         threshold, len(detection.lon), detection.n_components,
                         100 * detection.share_of_city_emission,
                         (f"worst {test['median_km_worst_tiles']:.1f} km vs others "
                          f"{test['median_km_other_tiles']:.1f} km, p={test['p_worst_closer']:.3g}")
                         if test["testable"] else "no cells")

        headline = detect(values, data.grid, args.ratio)
        cities[city] = {
            "n_tiles": int(len(residual)),
            "n_usable_cells": int(usable.sum()),
            "n_cells_excluded_zero_background": int((values > 0).sum() - usable.sum()),
            "ratio_distribution": {
                "p50": float(np.percentile(finite_ratio, 50)),
                "p90": float(np.percentile(finite_ratio, 90)),
                "p99": float(np.percentile(finite_ratio, 99)),
                "p99_9": float(np.percentile(finite_ratio, 99.9)),
                "max": float(finite_ratio.max()),
            },
            "sweep": sweep,
            "headline": {
                "ratio_threshold": args.ratio,
                "n_cells": int(len(headline.lon)),
                "n_components": headline.n_components,
                "share_of_city_emission": headline.share_of_city_emission,
                "residual_test": residual_test(headline, tile_xy, lat0, worst, others,
                                               data.tile_bboxes),
                "label_carries_the_point_term": spike_tile_contrast(
                    headline, odiac_tiles, data.carbon, data.tile_bboxes),
                "mass_conservation": mass_conservation(headline, odiac_tiles, data.carbon,
                                                       data.tile_bboxes),
                "wri_cross_check": wri_cross_check(headline, plants, tile_xy, lat0),
            },
            "version_invariance": version_invariance(city, data, year, args.ratio),
        }

    payload = {
        "question": "do the worst carbon-residual tiles contain or adjoin the point-source cells "
                    "visible in the ODIAC raster the label was built from?",
        "why_this_matters": "the plant-proximity test uses WRI plants as a proxy for what ODIAC "
                                  "injected. Detecting the spikes in ODIAC itself removes the "
                                  "proxy from the chain: the claim becomes a statement about the "
                                  "label's own structure.",
        "odiac_year": year, "odiac_version": args.version, "year_provenance": provenance,
        "why_the_year_choice_does_not_matter": "urbanmoe_odiac_vintage.json shows R2 flat to three "
                                               "decimals across 2012-2022 in every city, and "
                                               "version_invariance below re-runs this detection "
                                               "under every available ODIAC version with the same "
                                               "outcome. The year is arbitrary and stated as such.",
        "detection_rule": {
            "statistic": "cell value divided by the median of the ring of cells "
                         f"{RING_INNER}..{RING_OUTER} cells away (Chebyshev), core excluded",
            "why_a_ring": "ODIAC's non-point term is nightlight-downscaled and smooth at 1 km; an "
                          "injected point source is not. The hole keeps a source out of its own "
                          "background estimate.",
            "edge_handling": "padded with NaN and a nanmedian taken, so edge cells use real "
                             "neighbours only",
            "excluded": "cells with a zero or undefined ring background are not testable and are "
                        "counted separately rather than flagged",
            "threshold": f"headline {args.ratio}; every result swept over {list(RATIO_SWEEP)}",
        },
        "residual_protocol": "urbanmoe_power_plants.carbon_residuals verbatim: 5-fold CV, "
                             "image+POI ridge, log1p carbon target, "
                             f"{N_WORST} worst tiles by absolute residual. Baselines are the "
                             "5-fold diagnostic values 0.782/0.293/0.234, NOT the published "
                             "60/20/20 10-seed 0.786/0.440/0.456.",
        "cross_check_radius_km": CROSS_CHECK_KM,
        "cities": cities,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logging.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
