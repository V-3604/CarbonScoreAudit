"""Measure OSM POI coverage across the benchmark tiles, and cache each tile's bounding box.

The benchmark's own POI stream was not released, so the POI block used here is rebuilt from
OpenStreetMap. This script checks first that OSM has usable POI density where the tiles actually
are. Tiles cover whole metropolitan regions rather than urban cores, so sampling is stratified by
distance from each city's tile centroid; an unstratified draw is dominated by periphery.

It also writes the tile bounding-box index that the centroid lookup in common.py depends on,
which is why --dry-run is a useful first step even when the coverage question is settled.

Input : data/raw/urbanmoe/ (tile GeoTIFF headers)
Output: artifacts/reports/urbanmoe_tile_bboxes.json
        artifacts/reports/urbanmoe_poi_density.json

Run:
    python scripts/urbanmoe_poi_density.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/raw/urbanmoe"
REPORT_DIR = REPO_ROOT / "artifacts/reports"
OUT_JSON = REPORT_DIR / "urbanmoe_poi_density.json"
PARTIAL_JSON = REPORT_DIR / "urbanmoe_poi_density_partial.json"
BBOX_CACHE = REPORT_DIR / "urbanmoe_tile_bboxes.json"

SEED = 42
TILES_PER_CITY = 50
N_STRATA = 5
# The public API allows 2 slots per IP, each held for a query's execution plus a cooldown.
# A 1.0 s delay was rate-limited after roughly 20 queries, hence the more conservative default.
SLEEP_S = 25.0
MAX_CONSECUTIVE_FAILURES = 5
# Sending contact details is standard OSM etiquette; the default osmnx string is shared by every
# osmnx user and gets deprioritised. With it, /api/status returned 406; with a contact UA, 200.
# There is no default so that every run identifies whoever is actually sending the queries.
USER_AGENT_EXAMPLE = "CarbonScoreAudit/1.0 (your-institution; you@example.com)"
# The front domain redirects to z./lz4. subdomains and 504'd under the default UA while lz4
# answered 200 for the same query. Pinning removes that coin flip.
OVERPASS_URL = "https://lz4.overpass-api.de/api"

# The three cities ship under inconsistent directory names ("NewWork" holding "NewYork_file").
CITY_TILE_DIRS: dict[str, str] = {
    "beijing": "beijing/beijing_file",
    "shanghai": "shanghai/shanghai_sat_delete",
    "newyork": "NewWork/NewYork_file",
}

# UrbanMoE's POI categories are not published, so this is the standard OSM stand-in:
# the three keys that carry point-of-interest semantics for urban prediction.
POI_TAGS: dict[str, bool] = {"amenity": True, "shop": True, "public_transport": True}


def fix_proj_env() -> None:
    """Point PROJ at rasterio's own database.

    A PostgreSQL/PostGIS install on this machine puts its proj.db first on the search
    path, and rasterio then raises `CRSError: ... DATABASE.LAYOUT.VERSION.MINOR = 2` on
    the first reprojection. Setting both variables before the first CRS call avoids it.
    """
    import rasterio

    proj_data = Path(rasterio.__file__).parent / "proj_data"
    if proj_data.is_dir():
        os.environ["PROJ_LIB"] = str(proj_data)
        os.environ["PROJ_DATA"] = str(proj_data)


@dataclass(frozen=True)
class TileBox:
    """One tile's WGS84 bounding box and centroid."""

    city: str
    name: str
    west: float
    south: float
    east: float
    north: float

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.south + self.north) / 2.0, (self.west + self.east) / 2.0


def index_tiles(city: str, tile_dir: Path) -> list[TileBox]:
    """Read every GeoTIFF's bounds and reproject to EPSG:4326."""
    import rasterio
    from rasterio.warp import transform_bounds

    tiles = sorted(tile_dir.glob("*.tif"))
    if not tiles:
        raise SystemExit(f"No .tif tiles under {tile_dir}. Extract the UrbanMoE archive "
                         f"into {DATA_DIR} first (see urbanmoe_embed.py).")
    boxes: list[TileBox] = []
    for path in tiles:
        with rasterio.open(path) as src:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        boxes.append(TileBox(city, path.name, west, south, east, north))
    logging.info("%s: indexed %d tiles", city, len(boxes))
    return boxes


def load_or_build_index(refresh: bool) -> dict[str, list[TileBox]]:
    """Tile bboxes per city, cached to JSON (indexing 4,959 GeoTIFFs takes ~1 min)."""
    if BBOX_CACHE.exists() and not refresh:
        raw = json.loads(BBOX_CACHE.read_text(encoding="utf-8"))
        index = {city: [TileBox(**b) for b in boxes] for city, boxes in raw.items()}
        logging.info("Loaded tile index from %s (%d tiles)",
                     BBOX_CACHE.name, sum(len(v) for v in index.values()))
        return index

    fix_proj_env()
    index = {city: index_tiles(city, DATA_DIR / rel) for city, rel in CITY_TILE_DIRS.items()}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BBOX_CACHE.write_text(
        json.dumps({c: [asdict(b) for b in v] for c, v in index.items()}, indent=2),
        encoding="utf-8")
    logging.info("Wrote tile index to %s", BBOX_CACHE)
    return index


def stratified_sample(boxes: list[TileBox], n_tiles: int, n_strata: int,
                      rng: random.Random) -> list[tuple[TileBox, int]]:
    """Sample evenly across equal-count rings of distance from the city's median centre.

    Returns (tile, stratum) pairs; stratum 0 is the innermost ring. Equal-count rings
    (quantiles of distance) rather than equal-width ones, so every stratum has tiles to
    draw from even when a city's tiles are unevenly spread.
    """
    lats = np.array([b.centroid[0] for b in boxes])
    lons = np.array([b.centroid[1] for b in boxes])
    lat0, lon0 = float(np.median(lats)), float(np.median(lons))
    # Local equirectangular metres; exact enough for ranking tiles by distance.
    x = (lons - lon0) * 111_320.0 * np.cos(np.radians(lat0))
    y = (lats - lat0) * 110_540.0
    distance_m = np.hypot(x, y)

    order = np.argsort(distance_m)
    rings = np.array_split(order, n_strata)
    per_stratum = max(1, n_tiles // n_strata)

    sample: list[tuple[TileBox, int]] = []
    for stratum, ring in enumerate(rings):
        pick = rng.sample(list(ring), min(per_stratum, len(ring)))
        sample.extend((boxes[i], stratum) for i in pick)
    return sample


def count_pois(box: TileBox) -> int | None:
    """POIs inside one tile bbox, or None if the query itself failed.

    osmnx raises InsufficientResponseError for a genuinely empty area, which is a real
    zero and must not be confused with a network failure -- that distinction is the whole
    point of this gate, so the two cases are kept apart.
    """
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError

    try:
        features = ox.features_from_bbox(
            bbox=(box.west, box.south, box.east, box.north), tags=POI_TAGS)
    except InsufficientResponseError:
        return 0
    except Exception as exc:  # network / Overpass errors stay distinguishable from zero
        logging.warning("%s %s: query failed (%s)", box.city, box.name, type(exc).__name__)
        logging.debug("  %s", exc)
        return None
    return int(len(features))


def summarize(city: str, counts: list[int], strata: list[int], n_failed: int) -> dict:
    arr = np.array(counts, dtype=float)
    by_stratum = {}
    for stratum in sorted(set(strata)):
        vals = arr[np.array(strata) == stratum]
        by_stratum[f"ring_{stratum}"] = {
            "n_tiles": int(len(vals)),
            "median_pois": float(np.median(vals)),
            "mean_pois": round(float(vals.mean()), 2),
            "frac_empty": round(float((vals == 0).mean()), 3),
        }
    return {
        "city": city,
        "n_sampled": int(len(arr)),
        "n_failed_queries": n_failed,
        "median_pois": float(np.median(arr)),
        "mean_pois": round(float(arr.mean()), 2),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "max_pois": int(arr.max()),
        "frac_empty_tiles": round(float((arr == 0).mean()), 3),
        "by_distance_ring": by_stratum,
    }


def _write_partial(tile_rows: list[dict], per_city: list[dict], city: str,
                   counts: list[int], strata: list[int], n_failed: int) -> None:
    """Dump whatever was collected before an abort, so a stalled run is not wasted."""
    payload = {"aborted": True, "per_city": list(per_city), "tiles": tile_rows}
    if counts:
        payload["per_city"].append(summarize(city, counts, strata, n_failed))
    PARTIAL_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.warning("Wrote partial results to %s (%d tiles attempted).",
                    PARTIAL_JSON, len(tile_rows))


def verdict(per_city: list[dict], min_median: float) -> dict:
    """Apply the S18.4 decision rule to the measured distributions."""
    sparse = [c["city"] for c in per_city if c["median_pois"] < min_median]
    return {
        "rule": f"a city is POI-sparse if its median sampled tile holds < {min_median} POIs",
        "sparse_cities": sparse,
        "gate_passes": len(sparse) == 0,
        "reading": (
            "OSM POI coverage is too sparse in at least one city to substitute for the "
            "unreleased UrbanMoE POI stream; task 1 cannot separate method quality from "
            "OSM coverage there."
            if sparse else
            "OSM POI coverage is dense enough to rebuild a POI stream; task 1 is viable "
            "as an ablation. This POI block is rebuilt from OSM and is not the benchmark's own."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSM POI density gate for the UrbanMoE cities.")
    parser.add_argument("--tiles-per-city", type=int, default=TILES_PER_CITY)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap tiles per city (overrides --tiles-per-city); use for a pilot.")
    parser.add_argument("--n-strata", type=int, default=N_STRATA,
                        help="Equal-count distance rings to sample evenly across.")
    parser.add_argument("--sleep", type=float, default=SLEEP_S,
                        help="Seconds between Overpass queries. The public API allows 2 slots "
                             "per IP, each held for the query plus a cooldown, so anything "
                             "under ~20 s gets the IP cut off after a few dozen queries.")
    parser.add_argument("--user-agent", default=None,
                        help="Required unless --dry-run. Sent as ox.settings.http_user_agent, "
                             f"e.g. '{USER_AGENT_EXAMPLE}'. Include contact details; "
                             "the osmnx default is throttled and mirrors reject it outright.")
    parser.add_argument("--overpass-url", default=OVERPASS_URL,
                        help="Overpass endpoint. Pinned to the lz4 subdomain because the front "
                             "domain redirects unpredictably; mirrors also work, e.g. "
                             "https://overpass.kumi.systems/api")
    parser.add_argument("--max-consecutive-failures", type=int, default=MAX_CONSECUTIVE_FAILURES,
                        help="Abort once this many queries fail back to back, rather than "
                             "grinding through the whole sample collecting nothing.")
    parser.add_argument("--min-median-pois", type=float, default=5.0,
                        help="Median POIs per tile below which a city counts as sparse.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refresh-index", action="store_true",
                        help="Re-read every GeoTIFF instead of using the cached bbox index.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the index and print the sampling plan; no Overpass queries.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    n_tiles = args.limit if args.limit is not None else args.tiles_per_city
    rng = random.Random(args.seed)
    logging.info("seed=%d  tiles/city=%d  strata=%d  tags=%s",
                 args.seed, n_tiles, args.n_strata, sorted(POI_TAGS))

    index = load_or_build_index(args.refresh_index)
    plan = {city: stratified_sample(boxes, n_tiles, args.n_strata, rng)
            for city, boxes in index.items()}

    for city, sample in plan.items():
        lat, lon = np.mean([[b.centroid[0], b.centroid[1]] for b, _ in sample], axis=0)
        logging.info("%s: %d of %d tiles sampled, mean centroid (%.3f, %.3f)",
                     city, len(sample), len(index[city]), lat, lon)
    if args.dry_run:
        logging.info("Dry run: no Overpass queries issued.")
        return 0
    if args.user_agent is None:
        raise SystemExit("Pass --user-agent with your contact details, e.g. "
                         f"--user-agent \"{USER_AGENT_EXAMPLE}\". Overpass mirrors throttle or "
                         "reject the osmnx default.")

    import osmnx as ox
    ox.settings.overpass_rate_limit = True
    ox.settings.requests_timeout = 180
    ox.settings.http_user_agent = args.user_agent
    ox.settings.overpass_url = args.overpass_url
    logging.info("endpoint=%s  sleep=%.1fs  UA=%s",
                 args.overpass_url, args.sleep, args.user_agent)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_city, tile_rows = [], []
    for city, sample in plan.items():
        counts, strata, n_failed, consecutive = [], [], 0, 0
        for i, (box, stratum) in enumerate(sample, 1):
            n_pois = count_pois(box)
            if n_pois is None:
                n_failed += 1
                consecutive += 1
            else:
                counts.append(n_pois)
                strata.append(stratum)
                consecutive = 0
            tile_rows.append({"city": city, "tile": box.name, "ring": stratum,
                              "n_pois": n_pois,
                              "centroid_lat": round(box.centroid[0], 5),
                              "centroid_lon": round(box.centroid[1], 5)})
            if i % 10 == 0 or i == len(sample):
                logging.info("  %s %d/%d  running median %.1f POIs  (%d failed)",
                             city, i, len(sample),
                             float(np.median(counts)) if counts else float("nan"), n_failed)
            # Fail loudly. The first run ground through 26 dead queries in 45 minutes because
            # nothing stopped it; partial results are still written before raising.
            if consecutive >= args.max_consecutive_failures:
                _write_partial(tile_rows, per_city, city, counts, strata, n_failed)
                raise SystemExit(
                    f"{consecutive} consecutive query failures on {city} -- aborting. The "
                    f"endpoint is rate-limiting this client. "
                    f"Raise --sleep, switch --overpass-url to a mirror, or use the local-extract "
                    f"path. Partial results written to {PARTIAL_JSON.name}.")
            time.sleep(args.sleep)
        if not counts:
            raise SystemExit(f"Every Overpass query failed for {city}. Check the network "
                             f"before trusting any part of this gate.")
        per_city.append(summarize(city, counts, strata, n_failed))

    payload = {
        "meta": {
            "question": "is OSM POI coverage dense enough to substitute for UrbanMoE's "
                        "unreleased POI stream?",
            
            "seed": args.seed,
            "tiles_per_city": n_tiles,
            "n_strata": args.n_strata,
            "tags": sorted(POI_TAGS),
            "sampling": "equal-count distance rings from the city's median tile centre, "
                        "sampled evenly -- earlier one-off probes drew peripheral tiles "
                        "and were not representative",
            "caveat": "sparse OSM coverage is a data-availability finding about OSM, not a "
                      "statement about the cities; OSM contribution in China is legally "
                      "constrained. This POI block is rebuilt from OSM and is not the benchmark's own.",
        },
        "per_city": per_city,
        "verdict": verdict(per_city, args.min_median_pois),
        "tiles": tile_rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s", OUT_JSON)

    logging.info("==== POI density per city ====")
    logging.info("  %-10s %8s %8s %8s %10s", "city", "median", "mean", "max", "frac_empty")
    for entry in per_city:
        logging.info("  %-10s %8.1f %8.2f %8d %10.2f", entry["city"], entry["median_pois"],
                     entry["mean_pois"], entry["max_pois"], entry["frac_empty_tiles"])
    logging.info("==== verdict ====")
    logging.info("  gate_passes=%s  sparse=%s", payload["verdict"]["gate_passes"],
                 payload["verdict"]["sparse_cities"] or "none")
    logging.info("  %s", payload["verdict"]["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
