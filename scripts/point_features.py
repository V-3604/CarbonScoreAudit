"""Write a copy of each feature bundle with three point-source columns added to the POI block.

The columns are built from the cells flagged at K = 20 (ODIAC2024, 2015), for each tile:
  1. log1p of the flagged-cell emission inside the tile, area-weighted as in the label rebuild
  2. distance in km from the tile centre to the nearest flagged cell
  3. log1p of the flagged-cell emission within 5 km of the tile centre

These are inputs from which the label was built, so a model in use would not have them. The
bundles feed the sufficiency test in sufficiency_test.py.

Output: artifacts/features/{city}_odiacpoint.npz

Run:
    python scripts/point_features.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from common import CITIES, bundle_path, configure_logging, local_xy, tile_centres
from odiac import load_odiac_city
from point_sources import detect_city, point_mass_per_tile

VARIANT = "_odiacpoint"
NEAR_KM = 5.0


def point_columns(city: str) -> np.ndarray:
    data = load_odiac_city(city)
    values, detection = detect_city(data)
    centres = tile_centres(data.tile_bboxes)
    lat0 = float(centres[:, 1].mean())
    tile_xy = local_xy(centres[:, 0], centres[:, 1], lat0)
    cell_xy = local_xy(detection.lon, detection.lat, lat0)
    distance = np.sqrt(((tile_xy[:, None, :] - cell_xy[None, :, :]) ** 2).sum(axis=2))
    near_mass = np.where(distance <= NEAR_KM, detection.value[None, :], 0.0).sum(axis=1)
    return np.column_stack([np.log1p(point_mass_per_tile(values, detection, data)),
                            distance.min(axis=1),
                            np.log1p(near_mass)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    configure_logging()
    for city in CITIES:
        columns = point_columns(city)
        with np.load(bundle_path(city), allow_pickle=True) as bundle:
            arrays = {name: bundle[name] for name in bundle.files}
        if len(columns) != len(arrays["poi"]):
            raise SystemExit(f"{city}: {len(columns)} point-source rows for "
                             f"{len(arrays['poi'])} tiles.")
        arrays["poi"] = np.hstack([arrays["poi"], columns.astype(np.float32)])
        out_path = bundle_path(city, VARIANT)
        np.savez_compressed(out_path, **arrays)
        logging.info("%-9s %d tiles with point mass -> %s", city, int((columns[:, 0] > 0).sum()),
                     out_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
