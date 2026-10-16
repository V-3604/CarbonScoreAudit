"""Count OpenStreetMap points of interest per tile from local Geofabrik extracts.

The benchmark's own POI data was not released, so the POI block is rebuilt from OSM. Every node
and way tagged with amenity, shop or public_transport is kept; a way is placed at the mean of
its node locations. Each tile gets one count per value among the 20 most common amenity values,
10 most common shop values and 5 most common public_transport values across all three cities,
plus one total per key: 38 columns, so a column means the same thing in every city.

Input : data/raw/osm_extracts/china-latest.osm.pbf
        data/raw/osm_extracts/us-northeast-latest.osm.pbf
Output: artifacts/embeddings/poi_{city}.npy
        artifacts/embeddings/poi_{city}_files.npy
        artifacts/embeddings/poi_vocab.json

Run:
    python scripts/count_pois.py --dry-run
    python scripts/count_pois.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import osmium
from shapely import STRtree, box, points

from common import CITIES, EMBED_DIR, OSM_EXTRACT_DIR, TILE_INDEX, configure_logging

POI_KEYS = ("amenity", "shop", "public_transport")
TOP_VALUES_PER_KEY = {"amenity": 20, "shop": 10, "public_transport": 5}
CITY_EXTRACT = {
    "shanghai": "china-latest.osm.pbf",
    "beijing": "china-latest.osm.pbf",
    "newyork": "us-northeast-latest.osm.pbf",
}
EXTRACT_URL = {
    "china-latest.osm.pbf": "https://download.geofabrik.de/asia/china-latest.osm.pbf",
    "us-northeast-latest.osm.pbf":
        "https://download.geofabrik.de/north-america/us-northeast-latest.osm.pbf",
}

Bounds = tuple[float, float, float, float]


def scan_extract(path: Path, bounds: Bounds,
                 index_type: str) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """(lon, lat) and (key, value) of every POI inside `bounds`."""
    west, south, east, north = bounds
    coords: list[tuple[float, float]] = []
    kinds: list[tuple[str, str]] = []
    start_time = time.time()
    for obj in osmium.FileProcessor(str(path)).with_locations(index_type):
        key = next((k for k in POI_KEYS if k in obj.tags), None)
        if key is None:
            continue
        try:
            if obj.is_node():
                lon, lat = obj.location.lon, obj.location.lat
            elif obj.is_way():
                nodes = [(n.location.lon, n.location.lat) for n in obj.nodes
                         if n.location.valid()]
                if not nodes:
                    continue
                lon = float(np.mean([n[0] for n in nodes]))
                lat = float(np.mean([n[1] for n in nodes]))
            else:
                continue
        except (osmium.InvalidLocationError, RuntimeError):
            continue
        if west <= lon <= east and south <= lat <= north:
            coords.append((lon, lat))
            kinds.append((key, str(obj.tags[key])[:40]))
    logging.info("%s: %d POIs inside the tile envelope (%.0f s)", path.name, len(coords),
                 time.time() - start_time)
    return (np.asarray(coords) if coords else np.empty((0, 2))), kinds


def build_vocab(kinds: list[tuple[str, str]]) -> list[str]:
    vocab: list[str] = []
    for key, top_n in TOP_VALUES_PER_KEY.items():
        counter = Counter(value for k, value in kinds if k == key)
        vocab.extend(f"{key}={value}" for value, _ in counter.most_common(top_n))
    return vocab


def tile_counts(tiles: list[dict], coords: np.ndarray, kinds: list[tuple[str, str]],
                vocab: list[str]) -> np.ndarray:
    """(n_tiles, len(vocab) + len(POI_KEYS)) raw counts. A POI on a shared edge counts in both."""
    counts = np.zeros((len(tiles), len(vocab) + len(POI_KEYS)), dtype=np.float64)
    if len(coords) == 0:
        return counts
    tree = STRtree([box(t["west"], t["south"], t["east"], t["north"]) for t in tiles])
    poi_idx, tile_idx = tree.query(points(coords), predicate="intersects")
    column = {value: i for i, value in enumerate(vocab)}
    key_column = {key: len(vocab) + i for i, key in enumerate(POI_KEYS)}
    for p, t in zip(poi_idx, tile_idx):
        key, value = kinds[p]
        counts[t, key_column[key]] += 1
        if f"{key}={value}" in column:
            counts[t, column[f"{key}={value}"]] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-type", default="flex_mem",
                        help="pyosmium node-location index; flex_mem needs 2-4 GB RAM for the "
                             "China extract, sparse_file_array,<path> needs less")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the extracts and tile index, then exit")
    args = parser.parse_args()
    configure_logging()

    if not TILE_INDEX.exists():
        raise SystemExit(f"{TILE_INDEX} not found. Run scripts/build_tile_index.py first.")
    tiles = json.loads(TILE_INDEX.read_text(encoding="utf-8"))
    missing = [name for name in sorted(set(CITY_EXTRACT.values()))
               if not (OSM_EXTRACT_DIR / name).exists()]
    if missing:
        raise SystemExit(f"Missing OSM extracts in {OSM_EXTRACT_DIR}: "
                         + ", ".join(EXTRACT_URL[name] for name in missing))
    if args.dry_run:
        logging.info("extracts and tile index present; nothing parsed")
        return 0

    scanned = {}
    for extract in sorted(set(CITY_EXTRACT.values())):
        boxes = [t for city in CITIES if CITY_EXTRACT[city] == extract for t in tiles[city]]
        bounds = (min(t["west"] for t in boxes), min(t["south"] for t in boxes),
                  max(t["east"] for t in boxes), max(t["north"] for t in boxes))
        scanned[extract] = scan_extract(OSM_EXTRACT_DIR / extract, bounds, args.index_type)

    vocab = build_vocab([kind for _, kinds in scanned.values() for kind in kinds])
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        coords, kinds = scanned[CITY_EXTRACT[city]]
        counts = tile_counts(tiles[city], coords, kinds, vocab)
        np.save(EMBED_DIR / f"poi_{city}.npy", counts)
        np.save(EMBED_DIR / f"poi_{city}_files.npy",
                np.array([t["name"] for t in tiles[city]], dtype=object))
        logging.info("%-9s %d tiles x %d POI columns", city, *counts.shape)
    (EMBED_DIR / "poi_vocab.json").write_text(
        json.dumps({"vocab": vocab, "key_totals": list(POI_KEYS)}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
