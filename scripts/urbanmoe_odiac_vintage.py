"""Identify which ODIAC year and version the benchmark's carbon label was built from.

The benchmark does not state the vintage of its carbon values. This rebuilds the label from the
published ODIAC rasters for every candidate year, aggregates to the benchmark's 2 km tile grid and
regresses. The year whose R2 approaches 1 would identify the vintage; the slope identifies any
unit rescaling.

Source. ODIAC 1 km monthly GeoTIFFs read through the US Greenhouse Gas Center raster API
(earth.gov, CC-BY-4.0, no credentials), which crops server-side so city envelopes are fetched
rather than global rasters. That mirror was verified against the NIES original
(odiac2024_1km_excl_intl_1812.tif.gz) over the Beijing envelope: values are identical on every
valid cell, and the mirror's -9999 marks cells NIES stores as a genuine 0.0, which is why
NODATA_AS_ZERO is set.

Units are tonne carbon per cell per month, so a year is the sum of its twelve files and the
quantity is extensive. Aggregation to a tile weights each 1 km cell by the fraction of that cell
lying inside the tile, which assumes emissions are uniform within a cell. Both grids are
EPSG:4326 and the cos(latitude) factor cancels within a single cell, so overlap fractions are
computed in degrees and distances are never used here.

Every candidate year is reported whatever it shows. In practice the 2 km spatial pattern is
nearly invariant across 2012-2022, so no year is identifiable, which is the result reported in
the paper's appendix.

Input : ODIAC rasters (fetched and cached)
Output: artifacts/reports/urbanmoe_odiac_vintage.json
        data/external/odiac/crops/{version}/{city}_{YYYYMM}.npy

Run:
    python scripts/urbanmoe_odiac_vintage.py
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FEATURE_DIR = REPO_ROOT / "artifacts/features"
REPORT_DIR = REPO_ROOT / "artifacts/reports"
BBOX_JSON = REPORT_DIR / "urbanmoe_tile_bboxes.json"
CACHE_DIR = REPO_ROOT / "data/external/odiac/crops"

CITIES = ("shanghai", "beijing", "newyork")
CARBON = 0
RES_DEG = 1.0 / 120.0
NODATA_SENTINEL = -9999.0
RASTER_API = "https://earth.gov/ghgcenter/api/raster"

# ODIAC2025 is excluded on evidence, not preference: NIES released it 2026-06-05, five months
# after UrbanMoE appeared (arXiv 2601.22746, January 2026), so the benchmark cannot have used it.
# ODIAC2024 (released 2025-05-07) was the newest version available to them.
VERSION_COVERAGE = {"v2022": (2000, 2021), "v2023": (2000, 2022), "v2024": (2000, 2023)}
PRIMARY_VERSION = "v2024"
# Wider than the 2015-2021 window the imagery alone suggests: if the best-fitting year sat on an
# endpoint we could not tell a match from a truncated sweep. Shoulders on both sides make the
# identification readable.
CANDIDATE_YEARS = tuple(range(2012, 2023))
REQUEST_SLEEP_S = 1.0
MAX_RETRIES = 3


@dataclass(frozen=True)
class CropGrid:
    """An ODIAC crop covering one city, snapped out to 1/120-degree cell edges."""

    west: float
    south: float
    east: float
    north: float
    n_cols: int
    n_rows: int

    @property
    def bbox_str(self) -> str:
        return f"{self.west},{self.south},{self.east},{self.north}"


@dataclass(frozen=True)
class CityData:
    """The benchmark's own tiles and carbon labels, in bundle row order."""

    city: str
    tile_bboxes: np.ndarray  # (n_tiles, 4) as west, south, east, north
    carbon: np.ndarray
    grid: CropGrid


@dataclass(frozen=True)
class Fit:
    """Ordinary least squares of the benchmark label on the rebuilt ODIAC value."""

    r2: float
    r2_log1p: float
    slope: float
    intercept: float
    spearman: float
    median_ratio: float


def snap_to_grid(west: float, south: float, east: float, north: float) -> CropGrid:
    w = math.floor((west + 180.0) / RES_DEG) * RES_DEG - 180.0
    e = math.ceil((east + 180.0) / RES_DEG) * RES_DEG - 180.0
    s = math.floor((south + 90.0) / RES_DEG) * RES_DEG - 90.0
    n = math.ceil((north + 90.0) / RES_DEG) * RES_DEG - 90.0
    return CropGrid(w, s, e, n, int(round((e - w) / RES_DEG)), int(round((n - s) / RES_DEG)))


def load_tile_bboxes(city: str, files: np.ndarray) -> np.ndarray:
    """(n_tiles, 4) west/south/east/north in EPSG:4326, in the bundle's row order.

    Mirrors `urbanmoe_moe.load_centroids`, including its filename fix: the tile index names files
    'tile_N.tif' while embeddings and the label CSV use 'tile_N.tif.tif'.
    """
    if not BBOX_JSON.exists():
        raise SystemExit(f"{BBOX_JSON} not found. It is built by urbanmoe_poi_density.py "
                         f"--dry-run from the GeoTIFF headers.")
    raw = json.loads(BBOX_JSON.read_text(encoding="utf-8"))
    if city not in raw:
        raise SystemExit(f"no tile index for {city}; present: {sorted(raw)}")
    by_name = {(b["name"] if b["name"].endswith(".tif.tif") else b["name"] + ".tif"):
               (b["west"], b["south"], b["east"], b["north"])
               for b in raw[city]}
    try:
        return np.array([by_name[f] for f in files], dtype=np.float64)
    except KeyError as exc:
        raise SystemExit(f"{city}: tile {exc} has no bbox entry. Rebuild {BBOX_JSON.name}.") from exc


def load_city(city: str) -> CityData:
    bundle_path = FEATURE_DIR / f"urbanmoe_{city}.npz"
    if not bundle_path.exists():
        raise SystemExit(f"{bundle_path} not found. Run urbanmoe_build_features.py first.")
    bundle = np.load(bundle_path, allow_pickle=True)
    tile_bboxes = load_tile_bboxes(city, bundle["files"])
    grid = snap_to_grid(float(tile_bboxes[:, 0].min()), float(tile_bboxes[:, 1].min()),
                        float(tile_bboxes[:, 2].max()), float(tile_bboxes[:, 3].max()))
    return CityData(city, tile_bboxes, bundle["y"][:, CARBON].astype(np.float64), grid)


def item_id(version: str, year: int, month: int) -> str:
    """STAC item id. v2022 is named differently from v2023/v2024 on the same API."""
    stem = (f"{year}{month:02d}" if version == "v2022"
            else f"odiac{version[1:]}_1km_excl_intl_{year}{month:02d}")
    return f"odiac-ffco2-monthgrid-{version}-{stem}"


def cache_path(version: str, city: str, year: int, month: int) -> Path:
    return CACHE_DIR / version / f"{city}_{year}{month:02d}.npy"


def fetch_month(version: str, city: str, year: int, month: int, grid: CropGrid,
                sleep_s: float) -> np.ndarray:
    """One month of ODIAC over one city envelope, tonne C per cell, nodata mapped to zero."""
    path = cache_path(version, city, year, month)
    if path.exists():
        return np.load(path).astype(np.float64)

    url = (f"{RASTER_API}/collections/odiac-ffco2-monthgrid-{version}"
           f"/items/{item_id(version, year, month)}"
           f"/bbox/{grid.bbox_str}/{grid.n_cols}x{grid.n_rows}.tif?assets=co2-emissions")
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                blob = response.read()
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            logging.warning("  retry %d/%d for %s %s %d-%02d: %s",
                            attempt + 1, MAX_RETRIES, version, city, year, month, exc)
            time.sleep(2.0 * (attempt + 1))
    else:
        raise SystemExit(f"could not fetch ODIAC {version} {city} {year}-{month:02d} after "
                         f"{MAX_RETRIES} attempts: {last_error}\nURL: {url}")

    with rasterio.open(io.BytesIO(blob)) as dataset:
        values = dataset.read(1).astype(np.float64)
        valid = dataset.read(2) > 0
    if values.shape != (grid.n_rows, grid.n_cols):
        raise SystemExit(f"{city} {year}-{month:02d}: got {values.shape}, expected "
                         f"{(grid.n_rows, grid.n_cols)}. The raster API resampled the crop.")
    # Verified against the NIES original: the mirror's -9999 marks cells ODIAC stores as 0.0.
    values = np.where(valid, values, 0.0)
    if (values == NODATA_SENTINEL).any():
        raise SystemExit(f"{city} {year}-{month:02d}: {int((values == NODATA_SENTINEL).sum())} "
                         f"cells still hold {NODATA_SENTINEL} after masking; the mask band did "
                         f"not cover them and the nodata rule needs re-checking.")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.float32))  # source precision; summed in float64
    time.sleep(sleep_s)
    return values


def annual_grid(version: str, city: str, year: int, grid: CropGrid, sleep_s: float) -> np.ndarray:
    total = np.zeros((grid.n_rows, grid.n_cols), dtype=np.float64)
    for month in range(1, 13):
        total += fetch_month(version, city, year, month, grid, sleep_s)
    return total


def aggregate_to_tiles(values: np.ndarray, grid: CropGrid, tile_bboxes: np.ndarray) -> np.ndarray:
    """Area-weighted sum of an EXTENSIVE per-cell quantity onto the benchmark's 2 km tiles.

    Each 1 km cell contributes the fraction of ITS OWN area that falls inside the tile, which
    assumes emissions are spatially uniform within a cell. Row 0 of the crop is its north edge.
    """
    out = np.zeros(len(tile_bboxes), dtype=np.float64)
    weight_sums = np.zeros(len(tile_bboxes), dtype=np.float64)
    for i, (west, south, east, north) in enumerate(tile_bboxes):
        col_lo = max(0, math.floor((west - grid.west) / RES_DEG))
        col_hi = min(grid.n_cols, math.ceil((east - grid.west) / RES_DEG))
        row_lo = max(0, math.floor((grid.north - north) / RES_DEG))
        row_hi = min(grid.n_rows, math.ceil((grid.north - south) / RES_DEG))
        if col_lo >= col_hi or row_lo >= row_hi:
            raise SystemExit(f"tile {i} at [{west}, {south}, {east}, {north}] falls outside the "
                             f"downloaded crop; the envelope was computed wrongly")

        cols = np.arange(col_lo, col_hi)
        cell_west = grid.west + cols * RES_DEG
        frac_lon = (np.minimum(east, cell_west + RES_DEG) - np.maximum(west, cell_west)) / RES_DEG

        rows = np.arange(row_lo, row_hi)
        cell_north = grid.north - rows * RES_DEG
        frac_lat = (np.minimum(north, cell_north)
                    - np.maximum(south, cell_north - RES_DEG)) / RES_DEG

        weights = np.clip(frac_lat, 0.0, 1.0)[:, None] * np.clip(frac_lon, 0.0, 1.0)[None, :]
        out[i] = float((values[row_lo:row_hi, col_lo:col_hi] * weights).sum())
        weight_sums[i] = float(weights.sum())

    # The overlap fractions of a fully covered tile must sum to exactly tile_area / cell_area in
    # degree space. Catches misalignment, off-by-one indexing, and tiles clipped by the crop edge.
    # Note this is NOT 4: ODIAC's "1 km" grid is 1/120 degree square, which at Beijing's latitude
    # is 0.92 km north-south but 0.71 km east-west, so a 2 km tile spans about 6.3 cells.
    tile_areas = ((tile_bboxes[:, 2] - tile_bboxes[:, 0])
                  * (tile_bboxes[:, 3] - tile_bboxes[:, 1]))
    expected = tile_areas / (RES_DEG ** 2)
    worst = float(np.abs(weight_sums - expected).max())
    if worst > 1e-6 * float(expected.max()):
        raise SystemExit(f"cell weights do not sum to tile_area/cell_area: worst absolute "
                         f"discrepancy {worst:.3g} against an expected median of "
                         f"{np.median(expected):.4f}. The tile grid and the ODIAC grid are not "
                         f"aligned as assumed.")
    return out


def _ols_r2(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    r2 = 1.0 - float((residual ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return r2, float(slope), float(intercept)


def fit_label(carbon: np.ndarray, odiac: np.ndarray) -> Fit:
    """Linear fit in the original space, plus a log-space fit as the tail-robust companion.

    Carbon's skew reaches 15.4, so linear R2 is decided by a handful of extreme tiles and could
    pick a year on one tile alone. The log-space fit weights the whole distribution; agreement
    between the two is what makes an identification credible.
    """
    r2, slope, intercept = _ols_r2(carbon, odiac)
    r2_log, _, _ = _ols_r2(np.log1p(carbon), np.log1p(odiac))
    nonzero = odiac > 0
    return Fit(
        r2=r2,
        r2_log1p=r2_log,
        slope=float(slope),
        intercept=float(intercept),
        spearman=float(spearmanr(carbon, odiac).statistic),
        median_ratio=float(np.median(carbon[nonzero] / odiac[nonzero])) if nonzero.any() else float("nan"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="+", default=list(CITIES), choices=list(CITIES))
    parser.add_argument("--years", nargs="+", type=int, default=list(CANDIDATE_YEARS))
    parser.add_argument("--version", default=PRIMARY_VERSION, choices=sorted(VERSION_COVERAGE),
                        help="ODIAC version to sweep years within")
    parser.add_argument("--cross-check-versions", nargs="*", default=["v2022", "v2023"],
                        choices=sorted(VERSION_COVERAGE),
                        help="re-fit the best year under these versions too")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many NEW month-crop downloads (cached ones are free)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the download plan and exit without fetching")
    parser.add_argument("--sleep", type=float, default=REQUEST_SLEEP_S)
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "urbanmoe_odiac_vintage.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    cities = {city: load_city(city) for city in args.cities}
    for city, data in cities.items():
        logging.info("%-9s %4d tiles, crop %d x %d cells over [%.4f, %.4f, %.4f, %.4f]",
                     city, len(data.carbon), data.grid.n_cols, data.grid.n_rows,
                     data.grid.west, data.grid.south, data.grid.east, data.grid.north)

    years = sorted(set(args.years))
    for year in years:
        lo, hi = VERSION_COVERAGE[args.version]
        if not lo <= year <= hi:
            raise SystemExit(f"ODIAC {args.version} covers {lo}-{hi}; {year} is outside it")

    planned = [(args.version, city, year, month)
               for city in args.cities for year in years for month in range(1, 13)]
    missing = [job for job in planned if not cache_path(*job).exists()]
    logging.info("sweep: %d city-year-months, %d already cached, %d to download (~%.0f MB)",
                 len(planned), len(planned) - len(missing), len(missing), 0.15 * len(missing))
    if args.dry_run:
        logging.info("dry run: nothing fetched. Cross-check versions would add up to %d more "
                     "downloads for the winning year.", 12 * len(args.cities)
                     * len(args.cross_check_versions))
        return 0
    if args.limit is not None and len(missing) > args.limit:
        logging.warning("--limit %d against %d missing crops: the sweep stops at the first "
                        "city-year boundary past the limit and the report will be partial",
                        args.limit, len(missing))

    downloads = 0
    results: dict[str, dict] = {city: {"n_tiles": int(len(cities[city].carbon)), "years": {}}
                               for city in args.cities}
    stopped_early = False
    for city in args.cities:
        if stopped_early:
            break
        data = cities[city]
        for year in years:
            if args.limit is not None and downloads >= args.limit:
                logging.warning("--limit reached; stopping the sweep at %s %d. The report is "
                                "partial.", city, year)
                stopped_early = True
                break
            before = sum(1 for m in range(1, 13)
                         if not cache_path(args.version, city, year, m).exists())
            values = annual_grid(args.version, city, year, data.grid, args.sleep)
            downloads += before
            odiac = aggregate_to_tiles(values, data.grid, data.tile_bboxes)
            fit = fit_label(data.carbon, odiac)
            results[city]["years"][str(year)] = {
                "r2": fit.r2, "r2_log1p": fit.r2_log1p, "slope": fit.slope,
                "intercept": fit.intercept, "spearman": fit.spearman,
                "median_ratio_label_over_odiac": fit.median_ratio,
                "odiac_tile_sum_tonneC": float(odiac.sum()),
            }
            logging.info("  %-9s %d  R2 %8.5f  R2log %7.5f  rho %6.4f  slope %.6g",
                         city, year, fit.r2, fit.r2_log1p, fit.spearman, fit.slope)

    per_city_best = {}
    for city in args.cities:
        scored = results[city]["years"]
        if not scored:
            continue
        ranked = sorted(scored.items(), key=lambda kv: -kv[1]["r2"])
        best_year, best = ranked[0]
        runner_up_r2 = ranked[1][1]["r2"] if len(ranked) > 1 else float("nan")
        by_log = max(scored.items(), key=lambda kv: kv[1]["r2_log1p"])[0]
        by_rho = max(scored.items(), key=lambda kv: kv[1]["spearman"])[0]
        per_city_best[city] = {
            "best_year": int(best_year),
            "best_r2": best["r2"],
            "best_r2_log1p": best["r2_log1p"],
            "runner_up_year": int(ranked[1][0]) if len(ranked) > 1 else None,
            "runner_up_r2": runner_up_r2,
            "margin_over_runner_up": best["r2"] - runner_up_r2,
            "slope": best["slope"],
            "best_year_by_r2_log1p": int(by_log),
            "best_year_by_spearman": int(by_rho),
            "criteria_agree": bool(by_log == best_year == by_rho),
        }
        logging.info("%-9s best %s at R2 %.5f (runner-up %s at %.5f, margin %+.5f)",
                     city, best_year, best["r2"], ranked[1][0] if len(ranked) > 1 else "n/a",
                     runner_up_r2, best["r2"] - runner_up_r2)

    cross_check: dict[str, dict] = {}
    agreed_year = {b["best_year"] for b in per_city_best.values()}
    if stopped_early:
        cross_check["skipped"] = "the sweep stopped early on --limit, so no year is established"
    elif args.cross_check_versions and len(agreed_year) == 1:
        year = agreed_year.pop()
        for version in args.cross_check_versions:
            lo, hi = VERSION_COVERAGE[version]
            if not lo <= year <= hi:
                cross_check[version] = {"skipped": f"{version} covers {lo}-{hi}"}
                continue
            cross_check[version] = {}
            for city in args.cities:
                data = cities[city]
                values = annual_grid(version, city, year, data.grid, args.sleep)
                fit = fit_label(data.carbon, aggregate_to_tiles(values, data.grid,
                                                                data.tile_bboxes))
                cross_check[version][city] = {"year": year, "r2": fit.r2, "slope": fit.slope}
                logging.info("cross-check %s %-9s %d  R2 %.5f  slope %.6g",
                             version, city, year, fit.r2, fit.slope)
    elif args.cross_check_versions:
        cross_check["skipped"] = ("cities disagree on the best year, so there is no single year "
                                  "to cross-check across versions")

    best_r2_values = [b["best_r2"] for b in per_city_best.values()]
    identified = (not stopped_early and bool(best_r2_values) and min(best_r2_values) > 0.99
                  and len({b["best_year"] for b in per_city_best.values()}) == 1
                  and len(per_city_best) == len(args.cities))

    payload = {
        "question": "which ODIAC year and version is the benchmark's c_value?",
        "why": "The benchmark does not state the label vintage; its paper concedes this as an "
               "unresolvable limitation. Rebuilding the label from published rasters resolves it "
               "by measurement.",
        "source": "ODIAC 1 km monthly GeoTIFF via the US Greenhouse Gas Center raster API "
                  "(earth.gov, CC-BY-4.0). Verified bit-for-bit against the NIES original "
                  "odiac2024_1km_excl_intl_1812.tif.gz over the Beijing envelope on 2026-08-11; "
                  "the mirror's -9999 marks cells NIES stores as 0.0.",
        "method": {
            "units": "tonne carbon per 1 km cell per month; a year is the sum of its 12 files",
            "aggregation": "area-weighted sum onto the benchmark's 2 km tiles: each ODIAC cell "
                           "contributes the fraction of its own area inside the tile, which "
                           "assumes emissions are uniform within a cell",
            "crs": "EPSG:4326 throughout; overlap fractions computed in degrees, where the "
                   "cos(latitude) factor cancels within a cell. No distances are used.",
            "cell_geometry": "ODIAC's grid is 1/120 degree square, nominally 1 km at the equator "
                             "only; at Beijing's latitude a cell is about 0.92 km north-south and "
                             "0.71 km east-west, so a 2 km tile spans about 6.3 cells, not 4. "
                             "Weights are verified to sum to tile_area/cell_area exactly.",
            "nodata_rule": "mask band 0 -> value 0 (verified against NIES, not assumed)",
            "fit": "ordinary least squares of c_value on the aggregated ODIAC value, per city",
            "determinism": "no stochastic component; the result is exactly reproducible",
        },
        "version_swept": args.version,
        "versions_available": VERSION_COVERAGE,
        "odiac2025_excluded_because": "NIES released ODIAC2025 on 2026-06-05, after UrbanMoE "
                                      "appeared (arXiv 2601.22746, January 2026), so the "
                                      "benchmark cannot have used it.",
        "years_swept": years,
        "partial_run": stopped_early,
        "cities": results,
        "best_per_city": per_city_best,
        "cross_check_versions": cross_check,
        "vintage_identified": identified,
        "verdict": ("all cities agree on one year with R2 > 0.99" if identified else
                    "NOT identified cleanly: see best_per_city for the closest match per city and "
                    "its margin. Report this as measured, do not force a conclusion."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logging.info("wrote %s", args.out)
    logging.info("VERDICT: %s", payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
