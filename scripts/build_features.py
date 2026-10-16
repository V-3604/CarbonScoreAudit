"""Align image embeddings, POI counts and targets into one feature bundle per city.

Rows follow the image embeddings. POI counts are matched to them by tile name and log1p
transformed, and the three targets are read from the label CSV by the same names.

Output: artifacts/features/{city}.npz with arrays image, poi, y (c_value, pop_value,
        light_value) and files

Run:
    python scripts/build_features.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from common import (CITIES, EMBED_DIR, FEATURE_DIR, LABEL_CSV, RAW_DIR, TARGETS, bundle_path,
                    configure_logging, label_key)

EXPECTED_TILES = {"shanghai": 1558, "beijing": 2310, "newyork": 1091}


def load_block(stem: str, city: str) -> tuple[np.ndarray, np.ndarray]:
    values_path = EMBED_DIR / f"{stem}_{city}.npy"
    files_path = EMBED_DIR / f"{stem}_{city}_files.npy"
    for path in (values_path, files_path):
        if not path.exists():
            producer = "embed_images.py" if stem == "siglip" else "count_pois.py"
            raise SystemExit(f"{path} not found. Run scripts/{producer} first.")
    files = np.load(files_path, allow_pickle=True).astype(str)
    return np.load(values_path), np.array([label_key(name) for name in files])


def align(source: np.ndarray, target: np.ndarray, city: str, what: str) -> np.ndarray:
    """Row order that puts `source` names in the order of `target` names."""
    position = {name: i for i, name in enumerate(source)}
    try:
        return np.array([position[name] for name in target], dtype=int)
    except KeyError as exc:
        raise SystemExit(f"{city}: {what} has no row for tile {exc}.") from exc


def build_city(city: str) -> dict[str, np.ndarray]:
    image, files = load_block("siglip", city)
    poi, poi_files = load_block("poi", city)
    poi = np.log1p(poi[align(poi_files, files, city, "the POI block")])

    labels = pd.read_csv(RAW_DIR / LABEL_CSV[city]).set_index("filename")
    if len(labels) != len(files):
        raise SystemExit(f"{city}: the label CSV has {len(labels)} rows but {len(files)} tiles "
                         f"were embedded.")
    y = labels.loc[files, list(TARGETS)].to_numpy(dtype=np.float64)

    if len(np.unique(files)) != len(files):
        raise SystemExit(f"{city}: duplicate tile names, so every split would leak.")
    if len(files) != EXPECTED_TILES[city]:
        raise SystemExit(f"{city}: expected {EXPECTED_TILES[city]} tiles, found {len(files)}.")
    for name, block in (("image", image), ("poi", poi), ("y", y)):
        if not np.all(np.isfinite(block)):
            raise SystemExit(f"{city}: the {name} block holds non-finite values.")
    return {"image": image.astype(np.float32), "poi": poi.astype(np.float32), "y": y,
            "files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    configure_logging()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        arrays = build_city(city)
        np.savez_compressed(bundle_path(city), **arrays)
        logging.info("%-9s image %s, poi %s, y %s", city, arrays["image"].shape,
                     arrays["poi"].shape, arrays["y"].shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
