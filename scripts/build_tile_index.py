"""Record every benchmark tile's bounding box in EPSG:4326, read from the GeoTIFF headers.

Output: reports/tile_bboxes.json

Run:
    python scripts/build_tile_index.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from common import CITIES, RAW_DIR, REPO_ROOT, TILE_DIR, TILE_INDEX, configure_logging


@dataclass(frozen=True)
class TileBox:
    city: str
    name: str
    west: float
    south: float
    east: float
    north: float


def use_rasterio_proj_data() -> None:
    """Point PROJ at rasterio's bundled database in case another PROJ install comes first."""
    proj_data = Path(rasterio.__file__).parent / "proj_data"
    if proj_data.is_dir():
        os.environ["PROJ_LIB"] = str(proj_data)
        os.environ["PROJ_DATA"] = str(proj_data)


def index_city(city: str) -> list[TileBox]:
    tile_dir = RAW_DIR / TILE_DIR[city]
    paths = sorted(tile_dir.glob("*.tif"))
    if not paths:
        raise SystemExit(f"No .tif tiles under {tile_dir}. Extract the benchmark release into "
                         f"{RAW_DIR} first.")
    boxes = []
    for path in paths:
        with rasterio.open(path) as src:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        boxes.append(TileBox(city, path.name, west, south, east, north))
    logging.info("%-9s %d tiles", city, len(boxes))
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    configure_logging()
    use_rasterio_proj_data()
    index = {city: [asdict(box) for box in index_city(city)] for city in CITIES}
    TILE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    TILE_INDEX.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")
    logging.info("wrote %s", TILE_INDEX.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
