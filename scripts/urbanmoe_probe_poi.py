"""Ridge probe over the OSM-rebuilt POI block, alone and combined with imagery.

The benchmark's own POI stream was never released, so POI is rebuilt from OpenStreetMap here.
Arms:

  image            frozen SigLIP embeddings only
  poi              OSM POI category counts only
  image+poi        the combination used for every pipeline number in the paper
  image+text+poi   all three blocks

Same protocol as urbanmoe_probe.py: per-target ridge, StandardScaler fit on the training split
only, ten 60/20/20 seeds, R2 scored in the original target space.

Input : artifacts/embeddings/, artifacts/reports/urbanmoe_poi_density_local.json
Output: artifacts/reports/urbanmoe_probe_poi.json

Run:
    python scripts/urbanmoe_probe_poi.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from urbanmoe_probe import (
    CITIES, CITY_FOLDER, DATA_DIR, EMBED_DIR, REPORT_DIR,
    TARGETS, TARGET_LABEL, TABLE2_R2, TARGET_INDEX, probe_target,
)

OUT_JSON = REPORT_DIR / "urbanmoe_probe_poi_results.json"


def _align(files_from: np.ndarray, files_to: np.ndarray, what: str, city: str) -> np.ndarray:
    """Row order that puts `files_from` into `files_to` order. Loud on any mismatch."""
    pos = {f: i for i, f in enumerate(files_from)}
    try:
        return np.array([pos[f] for f in files_to], dtype=int)
    except KeyError as exc:
        raise SystemExit(
            f"{city}: {what} is missing tile {exc}. The tile index names files 'tile_N.tif' "
            f"while the embeddings and label CSV use 'tile_N.tif.tif' (a doubled extension in "
            f"their release); this join adds the suffix. Re-run "
            f"urbanmoe_poi_density_local.py --emit-features if the POI files are stale.") from exc


def load_city(city: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    img = np.load(EMBED_DIR / f"urbanmoe_siglip_{city}.npy").astype(np.float64)
    img_files = np.load(EMBED_DIR / f"urbanmoe_siglip_{city}_files.npy",
                        allow_pickle=True).astype(str)

    poi_path = EMBED_DIR / f"urbanmoe_poi_{city}.npy"
    if not poi_path.exists():
        raise SystemExit(f"{poi_path} not found. Run: urbanmoe_poi_density_local.py --emit-features")
    poi = np.load(poi_path).astype(np.float64)
    poi_files = np.load(EMBED_DIR / f"urbanmoe_poi_{city}_files.npy",
                        allow_pickle=True).astype(str)
    # Tile index uses 'tile_N.tif'; embeddings and CSV use 'tile_N.tif.tif'.
    poi_files = np.array([f if f.endswith(".tif.tif") else f + ".tif" for f in poi_files])
    poi = np.log1p(poi[_align(poi_files, img_files, "POI features", city)])

    # Each arm carries the widths of its concatenated blocks so probe_target can equalise their
    # share of the ridge penalty; single-block arms pass None and are unaffected.
    feature_sets: dict[str, tuple[np.ndarray, tuple[int, ...] | None]] = {
        "image": (img, None),
        "poi": (poi, None),
        "image+poi": (np.hstack([img, poi]), (img.shape[1], poi.shape[1])),
    }

    txt_path = EMBED_DIR / f"urbanmoe_siglip_text_{city}.npy"
    if txt_path.exists():
        txt = np.load(txt_path).astype(np.float64)
        txt_files = np.load(EMBED_DIR / f"urbanmoe_siglip_text_{city}_files.npy",
                            allow_pickle=True).astype(str)
        txt = txt[_align(txt_files, img_files, "text embeddings", city)]
        feature_sets["image+text+poi"] = (np.hstack([img, txt, poi]),
                                          (img.shape[1], txt.shape[1], poi.shape[1]))

    csv = pd.read_csv(DATA_DIR / CITY_FOLDER[city] / CITIES[city]).set_index("filename")
    labels = csv.loc[img_files].reset_index()
    return feature_sets, labels


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "meta": {
            "question": "does an OSM-rebuilt POI modality close the frozen-imagery gap to UrbanMoE?",
            "encoder": "siglip (frozen)",
            "probe": "ridge, single-task, per-target; StandardScaler(train); 10 seeds; 60/20/20",
            "poi_source": "OSM via Geofabrik extracts, category counts per 2 km tile, log1p",
            "gate2": "our OSM POI is NOT UrbanMoE's unreleased POI. Report as an ablation "
                     "('how much of the frozen-imagery gap is POI-shaped signal?'), never as a "
                     "leaderboard result. Always say 'with OSM-derived POI'.",
            "gap_closed_frac": "(image+poi - image) / (MoE - image)",
            "block_scaling": "concatenated arms divide each block by sqrt(its width) after standardisation, so 1152 image dims cannot regularise 38 POI dims away; the *__unbalanced entries keep the un-scaled value for comparison",
        },
        "cities": {},
    }

    summary_rows = []
    for city in list(CITIES):
        feats, labels = load_city(city)
        arms = list(feats)
        logging.info("=== %s (n=%d, poi dim=%d) arms: %s ===",
                     city, len(labels), feats["poi"][0].shape[1], ", ".join(arms))
        results["cities"][city] = {}
        for target in TARGETS:
            y = labels[target].to_numpy(dtype=np.float64)
            moe = TABLE2_R2[city]["UrbanMoE"][TARGET_INDEX[target]]
            cell: dict = {"urbanmoe_r2": moe,
                          "remoteclip_mt_r2": TABLE2_R2[city]["RemoteCLIP-MT"][TARGET_INDEX[target]]}
            for name in arms:
                mat, blocks = feats[name]
                stats = probe_target(mat, y, blocks)
                cell[name] = {k: stats[k] for k in ("r2_mean", "r2_std", "rmse_mean", "mae_mean")}
                if blocks:
                    # Unbalanced version kept so the artifact is documented, not just fixed.
                    raw = probe_target(mat, y)
                    cell[f"{name}__unbalanced"] = {"r2_mean": raw["r2_mean"]}

            r2_img = cell["image"]["r2_mean"]
            r2_ip = cell["image+poi"]["r2_mean"]
            gap = moe - r2_img
            cell["delta_poi_vs_image"] = round(r2_ip - r2_img, 4)
            cell["delta_image_poi_vs_moe"] = round(r2_ip - moe, 4)
            cell["gap_closed_frac"] = round((r2_ip - r2_img) / gap, 3) if gap > 1e-6 else None
            results["cities"][city][target] = cell
            summary_rows.append((city, target, r2_img, cell["poi"]["r2_mean"], r2_ip, moe,
                                 cell["gap_closed_frac"]))
            logging.info("  %-11s img %.3f | poi %.3f | img+poi %.3f | MoE %.3f | "
                         "Δ(+poi) %+.3f | gap closed %s",
                         TARGET_LABEL[target], r2_img, cell["poi"]["r2_mean"], r2_ip, moe,
                         r2_ip - r2_img,
                         "n/a" if cell["gap_closed_frac"] is None else f"{cell['gap_closed_frac']:.0%}")

    closed = [r[6] for r in summary_rows if r[6] is not None]
    beats = sum(1 for r in summary_rows if r[4] - r[5] > 0.02)
    ties = sum(1 for r in summary_rows if abs(r[4] - r[5]) <= 0.02)
    results["meta"]["headline"] = {
        "n_cells": len(summary_rows),
        "median_gap_closed_frac": round(float(np.median(closed)), 3) if closed else None,
        "cells_image_poi_beats_moe": beats,
        "cells_tie_moe": ties,
        "cells_below_moe": len(summary_rows) - beats - ties,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logging.info("Wrote %s", OUT_JSON)

    h = results["meta"]["headline"]
    logging.info("==== headline (OSM-derived POI; ablation, not a leaderboard comparison) ====")
    logging.info("  image+poi vs MoE over %d cells: %d beat, %d tie, %d below",
                 h["n_cells"], h["cells_image_poi_beats_moe"], h["cells_tie_moe"],
                 h["cells_below_moe"])
    logging.info("  median fraction of the image->MoE gap closed by POI: %s",
                 "n/a" if h["median_gap_closed_frac"] is None else f"{h['median_gap_closed_frac']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
