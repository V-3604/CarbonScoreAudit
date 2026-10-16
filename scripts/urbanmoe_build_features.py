"""Assemble the frozen modality blocks into one aligned, checked bundle per city.

Every analysis script reads these bundles rather than re-deriving the joins, so the alignment
happens once and is verified once.

The join has one footgun worth stating: the POI tile index names files `tile_N.tif` while the
embeddings and the label CSV use `tile_N.tif.tif`, a doubled extension in the released data. The
suffix fix and the alignment helper are imported from urbanmoe_probe_poi rather than
reimplemented, so the bundles carry exactly the rows the probe numbers are computed on.

Canonical row order is the image-embedding order. Text and POI are permuted into it, labels are
looked up by filename, and the result is asserted row for row before anything is written.

Input : artifacts/embeddings/, artifacts/reports/urbanmoe_poi_density_local.json
Output: artifacts/features/urbanmoe_{city}.npz

Run:
    python scripts/urbanmoe_build_features.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from urbanmoe_probe import CITIES, CITY_FOLDER, DATA_DIR, EMBED_DIR, REPO_ROOT, TARGETS
from urbanmoe_probe_poi import _align

FEATURE_DIR = REPO_ROOT / "artifacts/features"
MANIFEST_PATH = FEATURE_DIR / "urbanmoe_manifest.json"

# Row counts from the released zips, verified 2026-07-13. A mismatch means the extraction is
# incomplete or a stale embedding file is being picked up -- fail rather than train on it.
EXPECTED_ROWS = {"shanghai": 1558, "beijing": 2310, "newyork": 1091}


@dataclass
class Bundle:
    city: str
    files: np.ndarray
    image: np.ndarray
    text: np.ndarray
    poi: np.ndarray
    y: np.ndarray

    @property
    def block_widths(self) -> dict[str, int]:
        return {"image": self.image.shape[1], "text": self.text.shape[1], "poi": self.poi.shape[1]}


def _load_block(stem: str, city: str) -> tuple[np.ndarray, np.ndarray]:
    """(values, filenames) for one embedding block. Loud if either file is missing."""
    values_path = EMBED_DIR / f"urbanmoe_{stem}_{city}.npy"
    files_path = EMBED_DIR / f"urbanmoe_{stem}_{city}_files.npy"
    for path in (values_path, files_path):
        if not path.exists():
            raise SystemExit(
                f"{path} not found. Rebuild it: image/text via urbanmoe_embed.py and "
                f"urbanmoe_embed_text.py (GPU), POI via "
                f"urbanmoe_poi_density_local.py --emit-features (CPU).")
    return np.load(values_path), np.load(files_path, allow_pickle=True).astype(str)


def _check_finite(name: str, city: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        n_bad = int((~np.isfinite(values)).sum())
        raise SystemExit(f"{city}: {name} block holds {n_bad} non-finite values; refusing to write "
                         f"a bundle a model would silently train on.")


def build_city(city: str, with_osm_layers: bool = False,
               with_patch_max: bool = False) -> Bundle:
    image, image_files = _load_block("siglip", city)
    text, text_files = _load_block("siglip_text", city)
    poi, poi_files = _load_block("poi", city)

    # Their release doubles the extension on the embeddings and CSV but not on the tile index.
    poi_files = np.array([f if f.endswith(".tif.tif") else f + ".tif" for f in poi_files])

    text = text[_align(text_files, image_files, "text embeddings", city)]
    poi = poi[_align(poi_files, image_files, "POI features", city)]
    # Counts are heavily right-skewed (tile max 1453 against a median of 9-31); log1p matches
    # what urbanmoe_probe_poi.py feeds the published probe, so the bundles and the probe agree.
    poi = np.log1p(poi)

    if with_patch_max:
        # Max-over-patch-tokens pooling joins the IMAGE block, because it is an image feature and
        # the hypothesis is about pooling rather than about a new modality. The
        # attention-pooled block is kept alongside it so the two can be ablated apart.
        patch_max, patch_files = _load_block("siglipmax", city)
        patch_files = np.array([f if f.endswith(".tif.tif") else f + ".tif" for f in patch_files])
        patch_max = patch_max[_align(patch_files, image_files, "max-pooled patch embeddings", city)]
        image = np.hstack([image, patch_max])

    if with_osm_layers:
        # The ODIAC-motivated layers join the POI block rather than becoming a fourth
        # expert, so the architecture and the ablation ladder stay identical and the only thing
        # that changes between this variant and the default is the feature content.
        osm, osm_files = _load_block("osm", city)
        osm_files = np.array([f if f.endswith(".tif.tif") else f + ".tif" for f in osm_files])
        osm = osm[_align(osm_files, image_files, "OSM layer features", city)]
        poi = np.hstack([poi, np.log1p(osm)])

    csv = pd.read_csv(DATA_DIR / CITY_FOLDER[city] / CITIES[city]).set_index("filename")
    if len(csv) != len(image_files):
        raise SystemExit(f"{city}: label CSV has {len(csv)} rows but the image block has "
                         f"{len(image_files)}; the extraction is incomplete.")
    labels = csv.loc[image_files]
    y = labels[list(TARGETS)].to_numpy(dtype=np.float64)

    if len(np.unique(image_files)) != len(image_files):
        raise SystemExit(f"{city}: duplicate tile filenames in the image block; the join key is "
                         f"not unique and every downstream split would leak.")
    if len(image_files) != EXPECTED_ROWS[city]:
        raise SystemExit(f"{city}: expected {EXPECTED_ROWS[city]} tiles, found {len(image_files)}.")
    for name, block in (("image", image), ("text", text), ("poi", poi), ("y", y)):
        if len(block) != len(image_files):
            raise SystemExit(f"{city}: {name} has {len(block)} rows against {len(image_files)} tiles.")
        _check_finite(name, city, block)

    return Bundle(city=city, files=image_files,
                  image=image.astype(np.float32), text=text.astype(np.float32),
                  poi=poi.astype(np.float32), y=y)


def write_bundle(bundle: Bundle, suffix: str = "") -> dict:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURE_DIR / f"urbanmoe_{bundle.city}{suffix}.npz"
    arrays = {"image": bundle.image, "text": bundle.text, "poi": bundle.poi,
              "y": bundle.y, "files": bundle.files}
    np.savez_compressed(out_path, **arrays)

    entry = {
        "path": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_rows": int(len(bundle.files)),
        "targets": list(TARGETS),
        "block_widths": bundle.block_widths,
        "sha256": {name: hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
                   for name, arr in arrays.items() if name != "files"},
    }
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="+", default=list(CITIES))
    parser.add_argument("--with-osm-layers", action="store_true",
                        help="append the ODIAC-motivated OSM layers (urbanmoe_osm_layers.py) to "
                             "the POI block and write the _osmv2 variant, leaving the default "
                             "bundles -- and every number computed from them -- untouched")
    parser.add_argument("--with-patch-max", action="store_true",
                        help="append max-over-patch-token embeddings (urbanmoe_embed_patch.py, "
                             "GPU) to the IMAGE block and write the _patch variant")
    args = parser.parse_args()
    suffix = ("_osmv2" if args.with_osm_layers else "") + ("_patch" if args.with_patch_max else "")
    manifest_path = (MANIFEST_PATH if not suffix
                     else MANIFEST_PATH.with_name(f"urbanmoe_manifest{suffix}.json"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    manifest: dict = {
        "built": "urbanmoe_build_features.py",
        "canonical_row_order": "the image embedding order; text and POI are permuted into it",
        "poi_transform": "log1p (matches urbanmoe_probe_poi.py)",
        "note": "sha256 is over the raw bytes of each float array, truncated to 16 hex chars, "
                "for change detection only",
        "cities": {},
    }
    for city in args.cities:
        bundle = build_city(city, with_osm_layers=args.with_osm_layers,
                            with_patch_max=args.with_patch_max)
        entry = write_bundle(bundle, suffix)
        manifest["cities"][city] = entry
        logging.info("%-9s n=%-5d image %s  text %s  poi %s  y %s  -> %s",
                     city, entry["n_rows"], bundle.image.shape, bundle.text.shape,
                     bundle.poi.shape, bundle.y.shape, entry["path"])

    manifest["variant"] = suffix.lstrip("_") or "default"
    manifest["variant_blocks"] = {
        "poi": "38 base columns" + (" + ODIAC OSM layers" if args.with_osm_layers else ""),
        "image": "1152 attention-pooled SigLIP"
                 + (" + 1152 max-over-patch-tokens" if args.with_patch_max else ""),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(e["n_rows"] for e in manifest["cities"].values())
    logging.info("Wrote %s -- %d tiles across %d cities", manifest_path, total,
                 len(manifest["cities"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
