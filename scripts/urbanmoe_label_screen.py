"""Measure what removing tiles that hold injected point sources does to a reported score.

The detection rule in urbanmoe_odiac_pointsources.py finds the point cells ODIAC injects into its
raster. That rule needs no model and no training, so it can be run on a gridded product before
adopting it as a regression target. This script measures what it does to a reported score.

The removal is a reporting intervention, not a feature. The model is the ordinary image+POI ridge
on the default feature bundle and is never shown the point term, which is what separates this
from urbanmoe_plant_features.py. Nothing here is a method contribution and no score here is a
performance claim.

A tile is flagged when the area-weighted injected point mass inside it is greater than zero,
under the detection rule at K = 20. No model is retrained: probe_target returns the held-out
predictions for each of the ten folds, so the screened score is the identical model on the
identical fold with flagged tiles dropped from the test set only.

Removing tiles removes variance as well as error, so the error share and the variance share are
both reported; the score rises only where the first exceeds the second.

Predictions recorded before the first run:
  P-S1  The flagged tiles carry at least five times their tile share of squared error in Beijing
        and New York, and less than five times in Shanghai.
        FALSIFIED: Shanghai came in at 7.1 times.
  P-S2  The absolute change in carbon R2 is larger for Beijing than for Shanghai. Held.
  P-S3  No direction was predicted for the sign of the change.

Produces Table 2 of the paper.

Input : artifacts/features/urbanmoe_{city}.npz
        artifacts/reports/urbanmoe_odiac_pointsources.json
Output: artifacts/reports/urbanmoe_label_screen.json

Run:
    python scripts/urbanmoe_label_screen.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import r2_score

from common import FEATURE_DIR, load_city
from urbanmoe_probe import CITIES, REPORT_DIR, SEEDS, TARGETS, probe_target

OUT_JSON = REPORT_DIR / "urbanmoe_label_screen.json"
CARBON_INDEX = TARGETS.index("carbon") if "carbon" in TARGETS else 0
POINT_MASS_COLUMN = 0
SSE_SHARE_MULTIPLE = 5.0


def flagged_mask(city: str) -> np.ndarray:
    """Tiles holding area-weighted ODIAC injected point mass, from the odiacpoint POI block."""
    base_poi = np.load(FEATURE_DIR / f"urbanmoe_{city}.npz", allow_pickle=True)["poi"]
    point_poi = np.load(FEATURE_DIR / f"urbanmoe_{city}_odiacpoint.npz", allow_pickle=True)["poi"]
    if not np.allclose(point_poi[:, : base_poi.shape[1]], base_poi):
        raise SystemExit(
            f"{city}: the odiacpoint POI prefix no longer matches the default bundle, so the flag "
            f"cannot be read from it. Regenerate with urbanmoe_plant_features.py before rerunning."
        )
    extra = point_poi[:, base_poi.shape[1]:]
    return extra[:, POINT_MASS_COLUMN] > 0


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    n = len(diff)
    sigma = float(diff.std(ddof=1))
    t_stat = float(diff.mean() / (sigma / np.sqrt(n))) if sigma > 0 else float("nan")
    p = float(2 * student_t.sf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else float("nan")
    return {"mean_delta": float(diff.mean()), "paired_sigma": sigma, "t": t_stat, "p": p,
            "significant_at_0_05": bool(np.isfinite(p) and p < 0.05)}


def screen_city(city: str, seeds: tuple[int, ...]) -> dict:
    data = load_city(city, "")
    matrix = np.hstack([data.blocks["image"], data.blocks["poi"]])
    widths = (data.blocks["image"].shape[1], data.blocks["poi"].shape[1])
    y = data.y[:, CARBON_INDEX]
    flagged = flagged_mask(city)
    if len(flagged) != len(y):
        raise SystemExit(f"{city}: flag length {len(flagged)} does not match {len(y)} tiles.")

    stats = probe_target(matrix, y, block_sizes=widths, target_transform="log1p",
                         seeds=seeds, return_per_seed=True, return_predictions=True)

    r2_full, r2_screened, sse_share, sst_share, tile_share, n_flagged_fold = [], [], [], [], [], []
    for test_idx, pred in stats["fold_predictions"]:
        y_test = y[test_idx]
        is_flagged = flagged[test_idx]
        keep = ~is_flagged
        squared_error = (y_test - pred) ** 2
        centred = (y_test - y_test.mean()) ** 2

        r2_full.append(r2_score(y_test, pred))
        r2_screened.append(r2_score(y_test[keep], pred[keep]) if keep.sum() > 1 else float("nan"))
        sse_share.append(float(squared_error[is_flagged].sum() / squared_error.sum()))
        sst_share.append(float(centred[is_flagged].sum() / centred.sum()))
        tile_share.append(float(is_flagged.mean()))
        n_flagged_fold.append(int(is_flagged.sum()))

    r2_full = np.asarray(r2_full)
    r2_screened = np.asarray(r2_screened)
    sse_share_mean = float(np.mean(sse_share))
    tile_share_mean = float(np.mean(tile_share))
    concentration = sse_share_mean / tile_share_mean if tile_share_mean > 0 else float("nan")

    return {
        "n_tiles": int(len(y)),
        "n_flagged": int(flagged.sum()),
        "flagged_frac_of_city": float(flagged.mean()),
        "mean_flagged_per_test_fold": float(np.mean(n_flagged_fold)),
        "r2_full_mean": float(r2_full.mean()),
        "r2_full_per_seed": [float(v) for v in r2_full],
        "r2_screened_mean": float(r2_screened.mean()),
        "r2_screened_per_seed": [float(v) for v in r2_screened],
        "r2_paired": paired(r2_screened, r2_full),
        "sse_share_flagged_mean": sse_share_mean,
        "sst_share_flagged_mean": float(np.mean(sst_share)),
        "tile_share_flagged_mean": tile_share_mean,
        "sse_concentration_multiple": concentration,
        "meets_5x_sse_concentration": bool(concentration >= SSE_SHARE_MULTIPLE),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="+", default=list(CITIES))
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    seeds = tuple(SEEDS)

    results: dict = {
        "question": "does screening out tiles whose label carries an injected ODIAC point source "
                    "change what the benchmark reports for carbon?",
        "framing_constraint": "the screen is a REPORTING intervention on a model that never sees "
                              "the point term. No number here is a performance claim and none of "
                              "it belongs in a table against UrbanMoE.",
        "screen_rule": "flagged = area-weighted ODIAC injected point mass inside the tile > 0, "
                       "under the point-source detection rule (ratio K=20, ODIAC2024, year 2015)",
        "protocol": {
            "arm": "image+POI ridge on the DEFAULT bundle, carbon target, log1p transform, "
                   "R2 scored in the original carbon space",
            "split": "published 60/20/20, identical folds across both scores, no retraining",
            "seeds": list(seeds),
        },
        "prereg": {
            "written": "2026-08-29, before the first run; sealed in the module docstring",
            "P-S1": "flagged tiles carry >= 5x their tile share of test squared error in Beijing "
                    "and New York, and < 5x in Shanghai",
            "P-S2": "|delta R2| is larger for Beijing than for Shanghai",
            "P-S3": "no sign is predicted for delta R2; any directional reading is post hoc",
            "power_caveat": "17/13/29 flagged tiles of 1558/2310/1091, so ~3/3/6 per test fold; "
                            "the squared-error share is the stable statistic and a null on P-S2 "
                            "is a plausible outcome of sample size",
        },
        "cities": {},
    }

    for city in args.cities:
        logging.info("=== %s ===", city)
        cell = screen_city(city, seeds)
        results["cities"][city] = cell
        logging.info("  flagged %d/%d (%.2f%%), SSE share %.3f = %.1fx tile share, "
                     "R2 %.4f -> %.4f (delta %+.4f, p=%.4g)",
                     cell["n_flagged"], cell["n_tiles"], 100 * cell["flagged_frac_of_city"],
                     cell["sse_share_flagged_mean"], cell["sse_concentration_multiple"],
                     cell["r2_full_mean"], cell["r2_screened_mean"],
                     cell["r2_paired"]["mean_delta"], cell["r2_paired"]["p"])

    scored = results["cities"]
    if {"shanghai", "beijing", "newyork"} <= set(scored):
        bj, ny, sh = scored["beijing"], scored["newyork"], scored["shanghai"]
        results["prereg_outcome"] = {
            "P-S1_held": bool(bj["meets_5x_sse_concentration"]
                              and ny["meets_5x_sse_concentration"]
                              and not sh["meets_5x_sse_concentration"]),
            "P-S1_detail": {c: scored[c]["sse_concentration_multiple"] for c in scored},
            "P-S2_held": bool(abs(bj["r2_paired"]["mean_delta"])
                              > abs(sh["r2_paired"]["mean_delta"])),
            "P-S2_detail": {c: scored[c]["r2_paired"]["mean_delta"] for c in scored},
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    logging.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
