"""Count OSM POIs in every benchmark tile from a local OpenStreetMap extract.

Queries a downloaded extract rather than the Overpass API, which makes this a census of all 4,959
tiles instead of a sample: no sampling caveat, a dated snapshot anyone can re-download, and POI
tag definitions that can be varied freely.

Method
  1. Load the cached tile bounding boxes written by urbanmoe_poi_density.py.
  2. Stream each extract once with pyosmium, keeping nodes and ways carrying any POI key. Way
     positions are the mean of their node locations, which needs a node-location cache, hence
     --index-type (flex_mem is fastest, roughly 2-4 GB RAM for the China extract).
  3. Assign each POI to its tile with one vectorised shapely STRtree query.
  4. Report the per-city distribution and write the full per-tile counts.

Input : data/raw/osm_extracts/*.osm.pbf
        artifacts/reports/urbanmoe_tile_bboxes.json
Output: artifacts/reports/urbanmoe_poi_density_local.json

Run:
    python scripts/urbanmoe_poi_density_local.py --index-type flex_mem
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = REPO_ROOT / "data/raw/osm_extracts"
REPORT_DIR = REPO_ROOT / "artifacts/reports"
EMBED_DIR = REPO_ROOT / "artifacts/embeddings"
OUT_JSON = REPORT_DIR / "urbanmoe_poi_density_local.json"
BBOX_CACHE = REPORT_DIR / "urbanmoe_tile_bboxes.json"

# Same three keys the Overpass version used, so the two are comparable.
POI_KEYS = ("amenity", "shop", "public_transport")

# --emit-features turns the per-tile count into a per-tile CATEGORY vector, which is what a
# rebuilt POI modality actually needs (a single count carries almost no signal). Top-N values
# per key are chosen from the whole corpus so the vector means the same thing in every city.
TOP_VALUES_PER_KEY = {"amenity": 20, "shop": 10, "public_transport": 5}

CITY_EXTRACT: dict[str, str] = {
    "beijing": "china-latest.osm.pbf",
    "shanghai": "china-latest.osm.pbf",
    "newyork": "us-northeast-latest.osm.pbf",
}

MIN_MEDIAN_POIS = 5.0


@dataclass(frozen=True)
class TileBox:
    city: str
    name: str
    west: float
    south: float
    east: float
    north: float


def load_tiles(cities: list[str]) -> dict[str, list[TileBox]]:
    if not BBOX_CACHE.exists():
        raise SystemExit(
            f"{BBOX_CACHE} not found. Run urbanmoe_poi_density.py --dry-run first; it builds "
            f"the tile index from the GeoTIFF headers.")
    raw = json.loads(BBOX_CACHE.read_text(encoding="utf-8"))
    out = {c: [TileBox(**b) for b in raw[c]] for c in cities if c in raw}
    missing = [c for c in cities if c not in raw]
    if missing:
        raise SystemExit(f"No tile index for {missing}. Present: {sorted(raw)}")
    return out


def scan_extract(path: Path, bounds: tuple[float, float, float, float],
                 index_type: str) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """((lon, lat) per POI, [(key, value) per POI]) inside `bounds`. Nodes and ways.

    Ways are included because plenty of POIs are mapped as closed ways (a school building
    tagged amenity=school), and Overpass would have returned them -- excluding them would
    undercount relative to the sampled run this replaces. The (key, value) pairs are only
    used by --emit-features; the gate itself needs the count alone.
    """
    import osmium

    west, south, east, north = bounds
    lons: list[float] = []
    lats: list[float] = []
    kinds: list[tuple[str, str]] = []
    n_seen = 0
    t0 = time.time()

    processor = osmium.FileProcessor(str(path)).with_locations(index_type)
    for obj in processor:
        key = next((k for k in POI_KEYS if k in obj.tags), None)
        if key is None:
            continue
        n_seen += 1
        try:
            if obj.is_node():
                lon, lat = obj.location.lon, obj.location.lat
            elif obj.is_way():
                coords = [(n.location.lon, n.location.lat) for n in obj.nodes
                          if n.location.valid()]
                if not coords:
                    continue
                lon = float(np.mean([c[0] for c in coords]))
                lat = float(np.mean([c[1] for c in coords]))
            else:
                continue
        except (osmium.InvalidLocationError, RuntimeError):
            continue
        if west <= lon <= east and south <= lat <= north:
            lons.append(lon)
            lats.append(lat)
            kinds.append((key, str(obj.tags[key])[:40]))

    logging.info("  %s: %d tagged objects scanned, %d POIs inside the tile envelope (%.0f s)",
                 path.name, n_seen, len(lons), time.time() - t0)
    points = np.column_stack([np.asarray(lons), np.asarray(lats)]) if lons else np.empty((0, 2))
    return points, kinds


def assign_tiles(tiles: list[TileBox], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(poi_index, tile_index) pairs for every POI that lands in a tile, one STRtree query.

    Tiles can overlap slightly after reprojection from UTM, so a POI may land in more than
    one; both assignments are kept rather than arbitrarily picking a winner.
    """
    from shapely import STRtree, box, points as make_points

    if len(points) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    tree = STRtree([box(t.west, t.south, t.east, t.north) for t in tiles])
    return tree.query(make_points(points), predicate="intersects")


def counts_per_tile(n_tiles: int, tile_idx: np.ndarray) -> np.ndarray:
    counts = np.zeros(n_tiles, dtype=np.int64)
    if len(tile_idx):
        idx, n = np.unique(tile_idx, return_counts=True)
        counts[idx] = n
    return counts


def category_matrix(tiles: list[TileBox], poi_idx: np.ndarray, tile_idx: np.ndarray,
                    kinds: list[tuple[str, str]],
                    vocab: list[str]) -> np.ndarray:
    """Per-tile counts over `vocab` ("key=value"), plus one total column per key.

    Raw counts; the probe log1p-transforms them. Column order is `vocab` then POI_KEYS.
    """
    mat = np.zeros((len(tiles), len(vocab) + len(POI_KEYS)), dtype=np.float64)
    col_of = {v: i for i, v in enumerate(vocab)}
    key_col = {k: len(vocab) + i for i, k in enumerate(POI_KEYS)}
    for p, t in zip(poi_idx, tile_idx):
        key, value = kinds[p]
        mat[t, key_col[key]] += 1
        col = col_of.get(f"{key}={value}")
        if col is not None:
            mat[t, col] += 1
    return mat


def build_vocab(kinds: list[tuple[str, str]]) -> list[str]:
    """Top-N values per key across the whole corpus, so the vector means the same everywhere."""
    from collections import Counter

    vocab: list[str] = []
    for key, top_n in TOP_VALUES_PER_KEY.items():
        counter = Counter(v for k, v in kinds if k == key)
        vocab.extend(f"{key}={v}" for v, _ in counter.most_common(top_n))
    return vocab


def summarize(city: str, tiles: list[TileBox], counts: np.ndarray) -> dict:
    arr = counts.astype(float)
    lats = np.array([(t.south + t.north) / 2 for t in tiles])
    lons = np.array([(t.west + t.east) / 2 for t in tiles])
    lat0, lon0 = float(np.median(lats)), float(np.median(lons))
    x = (lons - lon0) * 111_320.0 * np.cos(np.radians(lat0))
    y = (lats - lat0) * 110_540.0
    distance_m = np.hypot(x, y)

    rings = np.array_split(np.argsort(distance_m), 5)
    by_ring = {}
    for i, ring in enumerate(rings):
        vals = arr[ring]
        by_ring[f"ring_{i}"] = {
            "n_tiles": int(len(vals)),
            "median_pois": float(np.median(vals)),
            "mean_pois": round(float(vals.mean()), 2),
            "frac_empty": round(float((vals == 0).mean()), 3),
        }
    return {
        "city": city,
        "n_tiles": int(len(arr)),
        "median_pois": float(np.median(arr)),
        "mean_pois": round(float(arr.mean()), 2),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "max_pois": int(arr.max()),
        "total_pois": int(arr.sum()),
        "frac_empty_tiles": round(float((arr == 0).mean()), 3),
        "by_distance_ring": by_ring,
    }


def verdict(per_city: list[dict], min_median: float) -> dict:
    sparse = [c["city"] for c in per_city if c["median_pois"] < min_median]
    return {
        "rule": f"a city is POI-sparse if its median tile holds < {min_median} POIs",
        "sparse_cities": sparse,
        "gate_passes": len(sparse) == 0,
        "reading": (
            "OSM POI coverage is too sparse in at least one city to substitute for the "
            "unreleased UrbanMoE POI stream; task 1 cannot separate method quality from OSM "
            "coverage there."
            if sparse else
            "OSM POI coverage is dense enough to rebuild a POI stream; task 1 is viable as an "
            "an ablation. This POI block is rebuilt from OSM and is not the benchmark's own."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count OSM POIs in every tile from a local extract.")
    parser.add_argument("--cities", nargs="+", default=sorted(CITY_EXTRACT),
                        choices=sorted(CITY_EXTRACT))
    parser.add_argument("--extract-dir", type=Path, default=EXTRACT_DIR)
    parser.add_argument("--index-type", default="flex_mem",
                        help="pyosmium node-location index. flex_mem is fastest (~2-4 GB RAM "
                             "for the China extract); use sparse_file_array,<path> if short "
                             "on memory.")
    parser.add_argument("--min-median-pois", type=float, default=MIN_MEDIAN_POIS)
    parser.add_argument("--emit-features", action="store_true",
                        help="Also write per-tile POI category vectors to "
                             "artifacts/embeddings/urbanmoe_poi_{city}.npy for the probe.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check the extracts exist and print the plan; parse nothing.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    tiles_by_city = load_tiles(args.cities)

    needed = {CITY_EXTRACT[c] for c in args.cities}
    missing = [f for f in sorted(needed) if not (args.extract_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"Missing extract(s) in {args.extract_dir}: {missing}\n"
            f"  https://download.geofabrik.de/asia/china-latest.osm.pbf\n"
            f"  https://download.geofabrik.de/north-america/us-northeast-latest.osm.pbf")

    for city in args.cities:
        tiles = tiles_by_city[city]
        logging.info("%s: %d tiles, extract %s", city, len(tiles), CITY_EXTRACT[city])
    if args.dry_run:
        logging.info("Dry run: extracts present, nothing parsed.")
        return 0

    # One pass per extract, shared across the cities that use it.
    by_extract: dict[str, list[str]] = {}
    for city in args.cities:
        by_extract.setdefault(CITY_EXTRACT[city], []).append(city)

    per_city, tile_rows = [], []
    scanned: dict[str, tuple] = {}
    for extract, cities in by_extract.items():
        all_tiles = [t for c in cities for t in tiles_by_city[c]]
        bounds = (min(t.west for t in all_tiles), min(t.south for t in all_tiles),
                  max(t.east for t in all_tiles), max(t.north for t in all_tiles))
        logging.info("Scanning %s for %s (envelope %.3f,%.3f .. %.3f,%.3f)",
                     extract, "+".join(cities), *bounds)
        scanned[extract] = scan_extract(args.extract_dir / extract, bounds, args.index_type)

    # Vocabulary is global across every extract so a column means the same thing in all cities.
    vocab = build_vocab([k for pts_kinds in scanned.values() for k in pts_kinds[1]]) \
        if args.emit_features else []
    if vocab:
        logging.info("POI vocabulary: %d values + %d key totals = %d features",
                     len(vocab), len(POI_KEYS), len(vocab) + len(POI_KEYS))

    for extract, cities in by_extract.items():
        points, kinds = scanned[extract]
        for city in cities:
            tiles = tiles_by_city[city]
            poi_idx, tile_idx = assign_tiles(tiles, points)
            counts = counts_per_tile(len(tiles), tile_idx)
            per_city.append(summarize(city, tiles, counts))
            tile_rows.extend({"city": city, "tile": t.name, "n_pois": int(n)}
                             for t, n in zip(tiles, counts))
            logging.info("  %s: median %.1f, mean %.1f, %.1f%% empty, %d POIs total",
                         city, per_city[-1]["median_pois"], per_city[-1]["mean_pois"],
                         100 * per_city[-1]["frac_empty_tiles"], per_city[-1]["total_pois"])
            if args.emit_features:
                mat = category_matrix(tiles, poi_idx, tile_idx, kinds, vocab)
                np.save(EMBED_DIR / f"urbanmoe_poi_{city}.npy", mat)
                np.save(EMBED_DIR / f"urbanmoe_poi_{city}_files.npy",
                        np.array([t.name for t in tiles], dtype=object))
                logging.info("    features %s -> %s", mat.shape,
                             (EMBED_DIR / f"urbanmoe_poi_{city}.npy").name)

    if args.emit_features:
        (EMBED_DIR / "urbanmoe_poi_vocab.json").write_text(
            json.dumps({"vocab": vocab, "key_totals": list(POI_KEYS)}, indent=2), encoding="utf-8")

    payload = {
        "meta": {
            "question": "is OSM POI coverage dense enough to substitute for UrbanMoE's "
                        "unreleased POI stream?",
            
            "method": "census of all tiles from local Geofabrik extracts; replaces the "
                      "rate-limited Overpass sample",
            "keys": list(POI_KEYS),
            "includes": "nodes and ways (way position = mean of its node locations)",
            "extracts": {c: CITY_EXTRACT[c] for c in args.cities},
            "caveat": "sparse OSM coverage is a data-availability finding about OSM, not a "
                      "statement about the cities; OSM contribution in China is legally "
                      "constrained. This POI block is rebuilt from OSM and is not the benchmark's own.",
        },
        "per_city": per_city,
        "verdict": verdict(per_city, args.min_median_pois),
        "tiles": tile_rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s", OUT_JSON)

    logging.info("==== POI density per city (ALL tiles) ====")
    logging.info("  %-10s %6s %8s %8s %8s %10s", "city", "tiles", "median", "mean", "max", "empty")
    for e in per_city:
        logging.info("  %-10s %6d %8.1f %8.2f %8d %9.1f%%", e["city"], e["n_tiles"],
                     e["median_pois"], e["mean_pois"], e["max_pois"],
                     100 * e["frac_empty_tiles"])
    logging.info("==== verdict ====")
    logging.info("  gate_passes=%s  sparse=%s", payload["verdict"]["gate_passes"],
                 payload["verdict"]["sparse_cities"] or "none")
    logging.info("  %s", payload["verdict"]["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
