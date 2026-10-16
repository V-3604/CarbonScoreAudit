"""Paths, constants and helpers shared by every script in this repository."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data/raw/urbanmoe"
OSM_EXTRACT_DIR = REPO_ROOT / "data/raw/osm_extracts"
PLANT_CSV = REPO_ROOT / "data/external/global_power_plant_database.csv"
ODIAC_CACHE_DIR = REPO_ROOT / "data/external/odiac/crops"
EMBED_DIR = REPO_ROOT / "artifacts/embeddings"
FEATURE_DIR = REPO_ROOT / "artifacts/features"
REPORT_DIR = REPO_ROOT / "reports"
FIGURE_DIR = REPO_ROOT / "figures"
TILE_INDEX = REPORT_DIR / "tile_bboxes.json"

CITIES = ("shanghai", "beijing", "newyork")
CITY_LABEL = {"shanghai": "Shanghai", "beijing": "Beijing", "newyork": "New York"}
# Paths inside the released benchmark archive, which names the New York folder "NewWork".
LABEL_CSV = {
    "shanghai": "shanghai/shanghai_merged_normalized.csv",
    "beijing": "beijing/Beijing_merged_normalized.csv",
    "newyork": "NewWork/NewYork_merged_normalized.csv",
}
TILE_DIR = {
    "shanghai": "shanghai/shanghai_sat_delete",
    "beijing": "beijing/beijing_file",
    "newyork": "NewWork/NewYork_file",
}
TARGETS = ("c_value", "pop_value", "light_value")
CARBON, LIGHT = 0, 2
SEEDS = tuple(range(41, 51))


@dataclass(frozen=True)
class Bundle:
    """One city's aligned features and targets, one row per tile."""

    files: np.ndarray
    image: np.ndarray
    poi: np.ndarray
    y: np.ndarray

    def image_poi(self) -> tuple[np.ndarray, tuple[int, int]]:
        """The image+POI feature matrix and its block widths."""
        return np.hstack([self.image, self.poi]), (self.image.shape[1], self.poi.shape[1])


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")


def label_key(tile_name: str) -> str:
    """The label CSV names tiles 'tile_N.tif.tif'; the GeoTIFFs on disk are 'tile_N.tif'."""
    return tile_name if tile_name.endswith(".tif.tif") else f"{tile_name}.tif"


def bundle_path(city: str, variant: str = "") -> Path:
    return FEATURE_DIR / f"{city}{variant}.npz"


def load_bundle(city: str, variant: str = "") -> Bundle:
    path = bundle_path(city, variant)
    if not path.exists():
        producer = "point_features.py" if variant else "build_features.py"
        raise SystemExit(f"{path} not found. Run scripts/{producer} first.")
    with np.load(path, allow_pickle=True) as data:
        bundle = Bundle(files=data["files"].astype(str),
                        image=data["image"].astype(np.float64),
                        poi=data["poi"].astype(np.float64),
                        y=data["y"].astype(np.float64))
    n_rows = len(bundle.files)
    if len(np.unique(bundle.files)) != n_rows:
        raise SystemExit(f"{path}: duplicate tile names, so every split would leak.")
    for name in ("image", "poi", "y"):
        if len(getattr(bundle, name)) != n_rows:
            raise SystemExit(f"{path}: {name} has {len(getattr(bundle, name))} rows for "
                             f"{n_rows} tiles.")
    return bundle


def load_tile_bboxes(city: str, files: np.ndarray) -> np.ndarray:
    """(n_tiles, 4) west, south, east, north in EPSG:4326, in the order of `files`."""
    if not TILE_INDEX.exists():
        raise SystemExit(f"{TILE_INDEX} not found. Run scripts/build_tile_index.py first.")
    index = json.loads(TILE_INDEX.read_text(encoding="utf-8"))
    by_name = {label_key(b["name"]): (b["west"], b["south"], b["east"], b["north"])
               for b in index[city]}
    try:
        return np.array([by_name[name] for name in files], dtype=np.float64)
    except KeyError as exc:
        raise SystemExit(f"{city}: tile {exc} is missing from {TILE_INDEX.name}. Rebuild it "
                         f"with scripts/build_tile_index.py.") from exc


def tile_centres(bboxes: np.ndarray) -> np.ndarray:
    """(n_tiles, 2) lon, lat of each tile's centre."""
    return np.column_stack([0.5 * (bboxes[:, 0] + bboxes[:, 2]),
                            0.5 * (bboxes[:, 1] + bboxes[:, 3])])


def local_xy(lon: np.ndarray, lat: np.ndarray, lat0: float) -> np.ndarray:
    """Equirectangular coordinates in km about latitude lat0, accurate at city scale."""
    return np.column_stack([lon * 111.320 * np.cos(np.radians(lat0)), lat * 110.540])


def nearest_km(from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
    """Distance in km from each point in `from_xy` to its nearest point in `to_xy`."""
    if len(to_xy) == 0:
        return np.full(len(from_xy), np.inf)
    distance = np.sqrt(((from_xy[:, None, :] - to_xy[None, :, :]) ** 2).sum(axis=2))
    return distance.min(axis=1)


def paired_t(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Two-sided paired t-test of a against b over matched seeds."""
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    sigma = float(diff.std(ddof=1))
    t_stat = float(diff.mean() / (sigma / np.sqrt(n))) if sigma > 0 else float("nan")
    p = float(2 * student_t.sf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else float("nan")
    return {"mean_delta": float(diff.mean()), "t": t_stat, "p": p}


def write_report(name: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    logging.info("wrote %s", path.relative_to(REPO_ROOT))
    return path
