"""Sufficiency test: does supplying the label's own point-source term close the carbon gap?

This is a diagnostic, not a method contribution. Any R2 it produces is not a performance result
and does not belong in a table against the benchmark's published numbers. The finding it supports
is that the only features which close the gap are inputs to the label's construction, which no
image contains and no deployed model would have.

Two variants, and the contrast between them is the point:

  plants      six columns from the WRI Global Power Plant Database, a modern public proxy for
              what ODIAC injected. Subject to a vintage mismatch: WRI is 2018-19 capacity in MW
              while the label's point term is frozen 2007 emissions with documented geolocation
              error.
  odiacpoint  three columns from the point-source cells detected in the ODIAC raster itself.
              This is the label's own term, with no proxy in the chain.

Both blocks join the POI block exactly as the OSM layers do, so the only thing that changes is
feature content. Block scaling is inherited from the probe and is not optional. One consequence
is stated rather than corrected: a 3-column block appended to 38 POI columns is scaled alongside
them, which if anything understates its effect.

Predictions recorded before the first run:
  P-B1  odiacpoint lifts Beijing carbon by at least +0.10 paired and does not lift Shanghai,
        because Beijing's label carries the point term in 7 of 7 source tiles while Shanghai's
        omits 8 of 10. Held.
  P-B2  plants gains less than odiacpoint on Beijing carbon, given the vintage mismatch. Held.

Input : artifacts/features/urbanmoe_{city}.npz
        artifacts/reports/urbanmoe_odiac_pointsources.json
        data/external/global_power_plant_database.csv
Output: artifacts/reports/urbanmoe_plant_features.json

Run:
    python scripts/urbanmoe_plant_features.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from urbanmoe_odiac_pointsources import DEFAULT_RATIO, detect  # noqa: E402
from urbanmoe_odiac_vintage import (  # noqa: E402
    CITIES,
    PRIMARY_VERSION,
    aggregate_to_tiles,
    annual_grid,
    load_city,
)
from urbanmoe_power_plants import (  # noqa: E402
    FOSSIL_FUELS,
    MIN_FOSSIL_MW,
    PLANT_CSV,
    local_xy,
    nearest_km,
    plants_near_tiles,
)

FEATURE_DIR = REPO_ROOT / "artifacts/features"
REPORT_DIR = REPO_ROOT / "artifacts/reports"

# Every column carries its one-line reason. No kitchen sink: six and three columns.
WRI_COLUMNS = {
    "dist_km_nearest_fossil50": "the plant-proximity statistic itself, encoded directly",
    "log1p_invdist_capacity": "sum of capacity_mw / (distance_km + 1); ODIAC's point term is "
                              "emissions-weighted and capacity is the public proxy for emissions",
    "log1p_count_2km": "sources within roughly one tile width, where the oracle ceiling "
                       "concentrates (92% of Beijing's error within 2 km)",
    "log1p_count_10km": "the radius the plant cross-check reports its fractions at",
    "log1p_max_capacity_5km": "a tile's point term is dominated by its single largest plant, not "
                              "by how many plants are nearby",
    "log1p_capacity_sum_5km": "the extensive companion to the max, since adjacent units at one "
                              "site are listed as separate plants",
}
ODIAC_POINT_COLUMNS = {
    "log1p_point_tonneC_in_tile": "the label's own injected mass inside the tile, area-weighted "
                                  "with the same rule as the rest of the forensics",
    "dist_km_nearest_point_cell": "distance to the nearest detected source, so tiles just outside "
                                  "one are distinguishable from tiles far from any",
    "log1p_point_tonneC_within_5km": "nearby injected mass; ODIAC's plant geolocation has "
                                     "documented error (Oda and Maksyutov 2011) so the exact cell "
                                     "may be offset from the true site",
}
NEAR_RADII_KM = {"count_2km": 2.0, "count_10km": 10.0, "capacity_5km": 5.0, "point_5km": 5.0}


def wri_block(city: str, tile_xy: np.ndarray, lat0: float,
              plants: pd.DataFrame) -> np.ndarray:
    """Six WRI-derived columns, in WRI_COLUMNS order."""
    in_region = plants_near_tiles(plants, tile_xy, lat0)
    fossil = in_region[in_region.primary_fuel.isin(FOSSIL_FUELS)
                       & (in_region.capacity_mw >= MIN_FOSSIL_MW)]
    if len(fossil) == 0:
        raise SystemExit(f"{city}: no fossil plants >= {MIN_FOSSIL_MW} MW near the tiles; the "
                         f"WRI block would be constant and the comparison meaningless.")

    plant_xy = local_xy(fossil.longitude.to_numpy(), fossil.latitude.to_numpy(), lat0)
    capacity = fossil.capacity_mw.to_numpy(dtype=np.float64)
    distance = np.sqrt(((tile_xy[:, None, :] - plant_xy[None, :, :]) ** 2).sum(axis=2))

    return np.column_stack([
        distance.min(axis=1),
        np.log1p((capacity[None, :] / (distance + 1.0)).sum(axis=1)),
        np.log1p((distance <= NEAR_RADII_KM["count_2km"]).sum(axis=1)),
        np.log1p((distance <= NEAR_RADII_KM["count_10km"]).sum(axis=1)),
        np.log1p(np.where(distance <= NEAR_RADII_KM["capacity_5km"], capacity[None, :], 0.0)
                 .max(axis=1)),
        np.log1p(np.where(distance <= NEAR_RADII_KM["capacity_5km"], capacity[None, :], 0.0)
                 .sum(axis=1)),
    ])


def odiac_point_block(city: str, data, tile_xy: np.ndarray, lat0: float, year: int,
                      version: str, ratio_threshold: float) -> tuple[np.ndarray, int]:
    """Three columns built from the point cells detected in ODIAC itself."""
    values = annual_grid(version, city, year, data.grid, sleep_s=0.0)
    detection = detect(values, data.grid, ratio_threshold)
    if len(detection.lon) == 0:
        raise SystemExit(f"{city}: no point cells detected at ratio {ratio_threshold}.")

    point_only = np.zeros_like(values)
    rows = np.round((data.grid.north - detection.lat) / (1.0 / 120.0) - 0.5).astype(int)
    cols = np.round((detection.lon - data.grid.west) / (1.0 / 120.0) - 0.5).astype(int)
    point_only[rows, cols] = values[rows, cols]
    in_tile = aggregate_to_tiles(point_only, data.grid, data.tile_bboxes)

    cell_xy = local_xy(detection.lon, detection.lat, lat0)
    distance = np.sqrt(((tile_xy[:, None, :] - cell_xy[None, :, :]) ** 2).sum(axis=2))
    within = np.where(distance <= NEAR_RADII_KM["point_5km"], detection.value[None, :], 0.0)

    return np.column_stack([
        np.log1p(in_tile),
        distance.min(axis=1),
        np.log1p(within.sum(axis=1)),
    ]), detection.n_components


def write_variant(city: str, suffix: str, extra: np.ndarray) -> dict:
    """Append `extra` to the POI block of the DEFAULT bundle and write a new file.

    The default bundle is opened read-only and never rewritten. A new suffix means a new filename,
    which is the repo's feature-variant contract.
    """
    source = FEATURE_DIR / f"urbanmoe_{city}.npz"
    out_path = FEATURE_DIR / f"urbanmoe_{city}{suffix}.npz"
    if out_path.resolve() == source.resolve():
        raise SystemExit(f"refusing to overwrite the default bundle {source}")
    with np.load(source, allow_pickle=True) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    if len(extra) != len(arrays["poi"]):
        raise SystemExit(f"{city}: extra block has {len(extra)} rows against "
                         f"{len(arrays['poi'])} tiles.")
    if not np.isfinite(extra).all():
        raise SystemExit(f"{city}: extra block holds non-finite values.")

    poi_width_before = int(arrays["poi"].shape[1])
    arrays["poi"] = np.hstack([arrays["poi"], extra.astype(np.float32)])
    np.savez_compressed(out_path, **arrays)
    return {"path": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "poi_width": f"{poi_width_before} -> {int(arrays['poi'].shape[1])}",
            "n_rows": int(len(extra))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="+", default=list(CITIES), choices=list(CITIES))
    parser.add_argument("--year", type=int, default=2015,
                        help="ODIAC year; the result is invariant across 2012-2022")
    parser.add_argument("--version", default=PRIMARY_VERSION)
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--plants-csv", type=Path, default=PLANT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "urbanmoe_plant_features.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.dry_run:
        logging.info("dry run: would write urbanmoe_{city}_plants.npz (%d cols) and "
                     "urbanmoe_{city}_odiacpoint.npz (%d cols) for %s; default bundles untouched",
                     len(WRI_COLUMNS), len(ODIAC_POINT_COLUMNS), ", ".join(args.cities))
        return 0
    if not args.plants_csv.exists():
        raise SystemExit(f"{args.plants_csv} not found.")

    plants = pd.read_csv(args.plants_csv, low_memory=False)
    report: dict[str, object] = {
        "purpose": "SUFFICIENCY TEST, a diagnostic and not a method contribution. See "
                   "This is a sufficiency test. No score here is a performance result.",
        "framing_constraint": "never describe a resulting R2 as a performance win, never place it "
                              "in a table against the benchmark's published numbers. "
                              "The finding is that the only feature closing the gap is the "
                              "label's own source, which no image can contain.",
        "preregistration": {
            "written": "before the first run",
            "P-B1": "odiacpoint lifts Beijing carbon >= +0.10 paired and does NOT lift Shanghai "
                    "(< +0.02); falsified if Shanghai gains >= +0.10 or Beijing gains < +0.10",
            "P-B2": "plants gains less than odiacpoint on Beijing carbon (vintage trap); "
                    "falsified if plants >= odiacpoint there",
            "P-B3": "New York gains from odiacpoint but less than Beijing; falsified if New "
                    "York's gain exceeds Beijing's",
        },
        "odiac_year": args.year, "odiac_version": args.version,
        "detection_ratio": args.ratio,
        "block_placement": "both blocks join the POI block, as the osmv2 layers do, so the "
                           "architecture and ladder are unchanged and block scaling by sqrt(width) "
                           "is inherited from the probe",
        "columns": {"plants": WRI_COLUMNS, "odiacpoint": ODIAC_POINT_COLUMNS},
        "cities": {},
    }

    for city in args.cities:
        data = load_city(city)
        lat0 = float(data.tile_bboxes[:, [1, 3]].mean())
        centre = np.column_stack([0.5 * (data.tile_bboxes[:, 0] + data.tile_bboxes[:, 2]),
                                  0.5 * (data.tile_bboxes[:, 1] + data.tile_bboxes[:, 3])])
        tile_xy = local_xy(centre[:, 0], centre[:, 1], lat0)

        wri = wri_block(city, tile_xy, lat0, plants)
        point, n_sources = odiac_point_block(city, data, tile_xy, lat0, args.year,
                                             args.version, args.ratio)
        report["cities"][city] = {
            "plants": write_variant(city, "_plants", wri),
            "odiacpoint": write_variant(city, "_odiacpoint", point),
            "n_detected_sources": n_sources,
            "median_dist_km_nearest_fossil50": float(np.median(wri[:, 0])),
            "median_dist_km_nearest_point_cell": float(np.median(point[:, 1])),
            "n_tiles_with_point_mass": int((point[:, 0] > 0).sum()),
        }
        logging.info("%-9s wrote both variants | %d detected sources | %d tiles hold point mass",
                     city, n_sources, int((point[:, 0] > 0).sum()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    logging.info("wrote %s", args.out)
    logging.info("now score with: urbanmoe_feature_ablation.py --test-features {plants,odiacpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
