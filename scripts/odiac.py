"""Download ODIAC, rebuild the benchmark's carbon label from it, and test which year it uses.

ODIAC 1 km monthly rasters are read through the US Greenhouse Gas Center raster API (earth.gov,
CC BY 4.0, no credentials), cropped to each city and cached under data/external/odiac/crops/.
Values are tonne carbon per cell per month, so a year is the sum of its twelve months. Each 2 km
tile receives every 1/120-degree cell it overlaps, weighted by the fraction of that cell's area
inside the tile.

The rebuild is compared with the released label for every year 2012-2022 under ODIAC2024, and
for 2015 under ODIAC2022, ODIAC2023 and ODIAC2024. These are the provenance numbers in Section 2
and the year-and-version paragraph of the appendix. ODIAC2025 was released after the benchmark,
so it is not considered.

Output: reports/odiac_vintage.json
        data/external/odiac/crops/{version}/{city}_{YYYYMM}.npy (cache, about 0.15 MB per file)

Run:
    python scripts/odiac.py --dry-run
    python scripts/odiac.py --limit 12
    python scripts/odiac.py
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr

from common import (CARBON, CITIES, ODIAC_CACHE_DIR, configure_logging, load_bundle,
                    load_tile_bboxes, write_report)

RASTER_API = "https://earth.gov/ghgcenter/api/raster"
CELL_DEG = 1.0 / 120.0
NODATA = -9999.0
VERSION_COVERAGE = {"v2022": (2000, 2021), "v2023": (2000, 2022), "v2024": (2000, 2023)}
ODIAC_VERSION = "v2024"
ODIAC_YEAR = 2015
SWEEP_YEARS = tuple(range(2012, 2023))
REQUEST_SLEEP_S = 1.0
MAX_RETRIES = 3


@dataclass(frozen=True)
class CropGrid:
    """A crop of the ODIAC grid around one city, snapped outward to cell edges."""

    west: float
    south: float
    east: float
    north: float
    n_cols: int
    n_rows: int


@dataclass(frozen=True)
class OdiacCity:
    """The benchmark tiles of one city, their carbon labels, and the crop that covers them."""

    city: str
    tile_bboxes: np.ndarray
    carbon: np.ndarray
    grid: CropGrid


def snap_to_grid(west: float, south: float, east: float, north: float) -> CropGrid:
    w = math.floor((west + 180.0) / CELL_DEG) * CELL_DEG - 180.0
    e = math.ceil((east + 180.0) / CELL_DEG) * CELL_DEG - 180.0
    s = math.floor((south + 90.0) / CELL_DEG) * CELL_DEG - 90.0
    n = math.ceil((north + 90.0) / CELL_DEG) * CELL_DEG - 90.0
    return CropGrid(w, s, e, n, int(round((e - w) / CELL_DEG)), int(round((n - s) / CELL_DEG)))


def load_odiac_city(city: str) -> OdiacCity:
    bundle = load_bundle(city)
    bboxes = load_tile_bboxes(city, bundle.files)
    grid = snap_to_grid(float(bboxes[:, 0].min()), float(bboxes[:, 1].min()),
                        float(bboxes[:, 2].max()), float(bboxes[:, 3].max()))
    return OdiacCity(city, bboxes, bundle.y[:, CARBON], grid)


def item_id(version: str, year: int, month: int) -> str:
    """STAC item id. ODIAC2022 uses a different naming scheme from 2023 and 2024."""
    stem = (f"{year}{month:02d}" if version == "v2022"
            else f"odiac{version[1:]}_1km_excl_intl_{year}{month:02d}")
    return f"odiac-ffco2-monthgrid-{version}-{stem}"


def cache_path(version: str, city: str, year: int, month: int) -> Path:
    return ODIAC_CACHE_DIR / version / f"{city}_{year}{month:02d}.npy"


def fetch_month(version: str, city: str, year: int, month: int, grid: CropGrid,
                sleep_s: float) -> np.ndarray:
    """One month of ODIAC over one city, tonne carbon per cell."""
    path = cache_path(version, city, year, month)
    if path.exists():
        return np.load(path).astype(np.float64)

    url = (f"{RASTER_API}/collections/odiac-ffco2-monthgrid-{version}"
           f"/items/{item_id(version, year, month)}"
           f"/bbox/{grid.west},{grid.south},{grid.east},{grid.north}"
           f"/{grid.n_cols}x{grid.n_rows}.tif?assets=co2-emissions")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                blob = response.read()
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES:
                raise SystemExit(f"Could not fetch ODIAC {version} {city} {year}-{month:02d} "
                                 f"after {MAX_RETRIES} attempts: {exc}\nURL: {url}") from exc
            logging.warning("retry %d/%d for %s %s %d-%02d: %s", attempt, MAX_RETRIES,
                            version, city, year, month, exc)
            time.sleep(2.0 * attempt)

    with rasterio.open(io.BytesIO(blob)) as dataset:
        values = dataset.read(1).astype(np.float64)
        valid = dataset.read(2) > 0
    if values.shape != (grid.n_rows, grid.n_cols):
        raise SystemExit(f"{city} {year}-{month:02d}: the API returned {values.shape}, expected "
                         f"{(grid.n_rows, grid.n_cols)}, so the crop was resampled.")
    # The API marks cells that the original NIES files store as 0.0 with -9999 and a zero mask.
    values = np.where(valid, values, 0.0)
    if (values == NODATA).any():
        raise SystemExit(f"{city} {year}-{month:02d}: cells outside the mask still hold "
                         f"{NODATA}; check the API's mask band.")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.float32))
    time.sleep(sleep_s)
    return values


def annual_grid(version: str, city: str, year: int, grid: CropGrid,
                sleep_s: float = REQUEST_SLEEP_S) -> np.ndarray:
    total = np.zeros((grid.n_rows, grid.n_cols), dtype=np.float64)
    for month in range(1, 13):
        total += fetch_month(version, city, year, month, grid, sleep_s)
    return total


def aggregate_to_tiles(values: np.ndarray, grid: CropGrid, tile_bboxes: np.ndarray) -> np.ndarray:
    """Sum each cell into the tiles it overlaps, weighted by its area fraction inside each tile.

    Overlap fractions are computed in degrees, where the cos(latitude) factor cancels within a
    cell. Row 0 of the crop is its northern edge.
    """
    out = np.zeros(len(tile_bboxes), dtype=np.float64)
    weight_sums = np.zeros(len(tile_bboxes), dtype=np.float64)
    for i, (west, south, east, north) in enumerate(tile_bboxes):
        col_lo = max(0, math.floor((west - grid.west) / CELL_DEG))
        col_hi = min(grid.n_cols, math.ceil((east - grid.west) / CELL_DEG))
        row_lo = max(0, math.floor((grid.north - north) / CELL_DEG))
        row_hi = min(grid.n_rows, math.ceil((grid.north - south) / CELL_DEG))
        if col_lo >= col_hi or row_lo >= row_hi:
            raise SystemExit(f"tile {i} at [{west}, {south}, {east}, {north}] lies outside "
                             f"the downloaded crop.")

        cell_west = grid.west + np.arange(col_lo, col_hi) * CELL_DEG
        frac_lon = (np.minimum(east, cell_west + CELL_DEG) - np.maximum(west, cell_west)) / CELL_DEG
        cell_north = grid.north - np.arange(row_lo, row_hi) * CELL_DEG
        frac_lat = (np.minimum(north, cell_north)
                    - np.maximum(south, cell_north - CELL_DEG)) / CELL_DEG

        weights = np.clip(frac_lat, 0.0, 1.0)[:, None] * np.clip(frac_lon, 0.0, 1.0)[None, :]
        out[i] = float((values[row_lo:row_hi, col_lo:col_hi] * weights).sum())
        weight_sums[i] = float(weights.sum())

    # A fully covered tile's weights sum to its area in cell units, about 6.3 cells for a 2 km
    # tile at Beijing's latitude. Any shortfall means the two grids are misaligned.
    expected = ((tile_bboxes[:, 2] - tile_bboxes[:, 0])
                * (tile_bboxes[:, 3] - tile_bboxes[:, 1])) / CELL_DEG ** 2
    worst = float(np.abs(weight_sums - expected).max())
    if worst > 1e-6 * float(expected.max()):
        raise SystemExit(f"Cell weights miss the tile area by up to {worst:.3g} cells; the tile "
                         f"grid and the ODIAC grid are not aligned.")
    return out


def label_match(carbon: np.ndarray, odiac_tiles: np.ndarray) -> dict[str, float]:
    """R2 of a least-squares line from the rebuild to the label, and their Spearman correlation."""
    slope, intercept = np.polyfit(odiac_tiles, carbon, 1)
    residual = carbon - (slope * odiac_tiles + intercept)
    r2 = 1.0 - float((residual ** 2).sum()) / float(((carbon - carbon.mean()) ** 2).sum())
    return {"r2": r2, "spearman": float(spearmanr(carbon, odiac_tiles).statistic)}


def year_sweep(data: OdiacCity, years: list[int], sleep_s: float) -> dict[str, object]:
    by_year = {}
    for year in years:
        values = annual_grid(ODIAC_VERSION, data.city, year, data.grid, sleep_s)
        by_year[year] = label_match(data.carbon, aggregate_to_tiles(values, data.grid,
                                                                    data.tile_bboxes))
        logging.info("  %-9s %d  R2 %.5f  Spearman %.4f", data.city, year,
                     by_year[year]["r2"], by_year[year]["spearman"])

    ranked = sorted(by_year, key=lambda y: -by_year[y]["r2"])
    best, runner_up = ranked[0], ranked[1]
    r2_values = [m["r2"] for m in by_year.values()]

    versions = {}
    for version, (first, last) in VERSION_COVERAGE.items():
        if first <= ODIAC_YEAR <= last:
            values = annual_grid(version, data.city, ODIAC_YEAR, data.grid, sleep_s)
            versions[version] = label_match(data.carbon, aggregate_to_tiles(values, data.grid,
                                                                            data.tile_bboxes))
    version_r2 = [m["r2"] for m in versions.values()]

    return {
        "n_tiles": int(len(data.carbon)),
        "years": {str(y): by_year[y] for y in years},
        "best_year": int(best),
        "runner_up_year": int(runner_up),
        "margin_over_runner_up": by_year[best]["r2"] - by_year[runner_up]["r2"],
        "spearman_at_best_year": by_year[best]["spearman"],
        "r2_range_over_years": max(r2_values) - min(r2_values),
        f"versions_at_{ODIAC_YEAR}": versions,
        "r2_range_over_versions": max(version_r2) - min(version_r2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", type=int, default=list(SWEEP_YEARS))
    parser.add_argument("--limit", type=int, default=None,
                        help="download at most this many missing month crops, then stop")
    parser.add_argument("--dry-run", action="store_true",
                        help="report how many month crops are missing and exit")
    parser.add_argument("--sleep", type=float, default=REQUEST_SLEEP_S,
                        help="seconds between API requests")
    args = parser.parse_args()
    configure_logging()

    years = sorted(set(args.years))
    first, last = VERSION_COVERAGE[ODIAC_VERSION]
    if not all(first <= y <= last for y in years):
        raise SystemExit(f"ODIAC {ODIAC_VERSION} covers {first}-{last}; got {years}.")

    cities = {city: load_odiac_city(city) for city in CITIES}
    jobs = sorted({(ODIAC_VERSION, city, year, month)
                   for city in CITIES for year in years for month in range(1, 13)}
                  | {(version, city, ODIAC_YEAR, month)
                     for version in VERSION_COVERAGE for city in CITIES
                     for month in range(1, 13)})
    missing = [job for job in jobs if not cache_path(*job).exists()]
    logging.info("%d month crops needed, %d cached, %d to download (about %.0f MB, %.0f min)",
                 len(jobs), len(jobs) - len(missing), len(missing), 0.15 * len(missing),
                 len(missing) * (args.sleep + 2.0) / 60)
    if args.dry_run:
        return 0
    if args.limit is not None and len(missing) > args.limit:
        for version, city, year, month in missing[:args.limit]:
            fetch_month(version, city, year, month, cities[city].grid, args.sleep)
        logging.info("downloaded %d of %d missing crops; run again without --limit to finish",
                     args.limit, len(missing))
        return 0

    report = {
        "odiac_version": ODIAC_VERSION,
        "years": years,
        "version_check_year": ODIAC_YEAR,
        "match": "least-squares R2 and Spearman correlation between the released carbon label "
                 "and the rebuild, per city",
        "cities": {city: year_sweep(data, years, args.sleep) for city, data in cities.items()},
    }
    write_report("odiac_vintage.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
