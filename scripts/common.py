"""Shared paths and feature-bundle loading for the ODIAC label audit.

Every analysis script reads the same per-city feature bundle written by the embedding and POI
steps, so the loader and the tile-centroid lookup live here rather than being duplicated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FEATURE_DIR = REPO_ROOT / "artifacts/features"
REPORT_DIR = REPO_ROOT / "artifacts/reports"
EMBED_DIR = REPO_ROOT / "artifacts/embeddings"
BBOX_JSON = REPORT_DIR / "urbanmoe_tile_bboxes.json"

CITIES = ("shanghai", "beijing", "newyork")
TARGETS = ("carbon", "population", "light")
MODALITIES = ("image", "text", "poi")


@dataclass
class CityData:
    city: str
    files: np.ndarray
    blocks: dict[str, np.ndarray]
    y: np.ndarray

    @property
    def n_rows(self) -> int:
        return len(self.files)


def load_city(city: str, variant: str = "") -> CityData:
    """Load one city's feature bundle, checking the invariants every split depends on."""
    path = FEATURE_DIR / f"urbanmoe_{city}{variant}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/urbanmoe_build_features.py first.")
    with np.load(path, allow_pickle=True) as bundle:
        blocks = {name: bundle[name].astype(np.float64) for name in MODALITIES}
        data = CityData(city=city, files=bundle["files"].astype(str), blocks=blocks,
                        y=bundle["y"].astype(np.float64))
    if len(np.unique(data.files)) != data.n_rows:
        raise SystemExit(f"{city}: duplicate tile keys in the bundle; every split would leak.")
    for name, block in blocks.items():
        if len(block) != data.n_rows:
            raise SystemExit(f"{city}: {name} has {len(block)} rows against {data.n_rows} tiles.")
    return data


def load_centroids(city: str, files: np.ndarray) -> np.ndarray:
    """(lon, lat) tile centroids in the bundle's row order."""
    if not BBOX_JSON.exists():
        raise SystemExit(f"{BBOX_JSON} not found. Build it with scripts/urbanmoe_poi_density.py "
                         f"--dry-run, which reads the tile GeoTIFF headers.")
    raw = json.loads(BBOX_JSON.read_text(encoding="utf-8"))
    if city not in raw:
        raise SystemExit(f"no tile index for {city}; present: {sorted(raw)}")
    # The tile index names files 'tile_N.tif'; embeddings and the label CSV use 'tile_N.tif.tif'.
    by_name = {(b["name"] if b["name"].endswith(".tif.tif") else b["name"] + ".tif"):
               (0.5 * (b["west"] + b["east"]), 0.5 * (b["south"] + b["north"]))
               for b in raw[city]}
    try:
        return np.array([by_name[f] for f in files], dtype=np.float64)
    except KeyError as exc:
        raise SystemExit(f"{city}: tile {exc} has no bbox entry. Rebuild {BBOX_JSON.name}.") from exc
