"""How much of the carbon score is reachable from the nightlight target alone.

ODIAC places point sources from a power-plant inventory and spreads everything else using
satellite nighttime lights. Nighttime light is itself one of this benchmark's three targets, so
carbon and light are coupled by construction rather than by urban coincidence.

A linear fit understates that coupling: the relationship is close to monotone and strongly
non-linear, and carbon's skew is what breaks a linear read. The arms below therefore use monotone
parameterisations, scored held out on the same folds and seeds as the pipeline.

  light_isotonic     carbon predicted from the light target alone, isotonic regression
  light_loglinear    carbon predicted from the light target alone, ridge on log1p of both
  image_poi          the image+POI pipeline, for reference
  image_poi_light    that pipeline plus the light target, as a ceiling

The light target is a target, never a legitimate model input. These arms characterise the label
and are not benchmark results. Produces Table 1 of the paper.

Input : artifacts/features/urbanmoe_{city}.npz
Output: artifacts/reports/urbanmoe_carbon_from_light.json

Run:
    python scripts/urbanmoe_carbon_from_light.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from urbanmoe_probe import probe_target, split_60_20_20  # noqa: E402

FEATURE_DIR = REPO_ROOT / "artifacts/features"
REPORT_DIR = REPO_ROOT / "artifacts/reports"

CITIES = ("shanghai", "beijing", "newyork")
CITY_LABEL = {"shanghai": "Shanghai", "beijing": "Beijing", "newyork": "New York"}
SEEDS = tuple(range(41, 51))
CARBON, POPULATION, LIGHT = 0, 1, 2


def isotonic_from_light(light: np.ndarray, carbon: np.ndarray,
                        splits: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    """Carbon predicted by the best monotone function of light alone, fit on train only."""
    r2s: list[float] = []
    residual_sum = np.zeros(len(carbon))
    residual_count = np.zeros(len(carbon))
    for train, test in splits:
        model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        model.fit(light[train], carbon[train])
        pred = model.predict(light[test])
        r2s.append(float(r2_score(carbon[test], pred)))
        residual_sum[test] += carbon[test] - pred
        residual_count[test] += 1
    seen = residual_count > 0
    residual = np.full(len(carbon), np.nan)
    residual[seen] = residual_sum[seen] / residual_count[seen]
    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s, ddof=1)),
        "r2_per_seed": [float(v) for v in r2s],
        "residual": residual,
    }


def analyse_city(city: str) -> dict[str, object]:
    bundle = np.load(FEATURE_DIR / f"urbanmoe_{city}.npz", allow_pickle=True)
    image, poi, y = bundle["image"], bundle["poi"], bundle["y"]
    carbon, light = y[:, CARBON], y[:, LIGHT]
    splits = [split_60_20_20(len(carbon), s) for s in SEEDS]

    light_col = light.reshape(-1, 1)
    isotonic = isotonic_from_light(light, carbon, splits)
    loglinear = probe_target(np.log1p(light_col), carbon, target_transform="log1p",
                             splits=splits, return_per_seed=True)
    image_poi = probe_target(np.hstack([image, poi]), carbon,
                             block_sizes=(image.shape[1], poi.shape[1]),
                             target_transform="log1p", splits=splits, return_per_seed=True)
    image_poi_light = probe_target(np.hstack([image, poi, light_col]), carbon,
                                   block_sizes=(image.shape[1], poi.shape[1], 1),
                                   target_transform="log1p", splits=splits,
                                   return_per_seed=True)

    # Does the benchmark model fail on exactly the tiles light cannot explain?
    baseline = probe_target(np.hstack([image, poi]), carbon,
                            block_sizes=(image.shape[1], poi.shape[1]),
                            target_transform="log1p", splits=splits, return_predictions=True)
    model_resid_sum = np.zeros(len(carbon))
    model_resid_count = np.zeros(len(carbon))
    for test, pred in baseline["fold_predictions"]:
        model_resid_sum[test] += carbon[test] - pred
        model_resid_count[test] += 1
    seen = model_resid_count > 0
    model_under = np.full(len(carbon), np.nan)
    model_under[seen] = model_resid_sum[seen] / model_resid_count[seen]

    light_resid = np.asarray(isotonic["residual"])
    both = seen & np.isfinite(light_resid)
    rho, p_value = spearmanr(light_resid[both], model_under[both])

    top_decile = carbon >= np.quantile(carbon, 0.9)
    rest = ~top_decile
    return {
        "city": city,
        "n_tiles": int(len(carbon)),
        "arms": {
            "light_isotonic": {k: isotonic[k] for k in ("r2_mean", "r2_std", "r2_per_seed")},
            "light_loglinear": {k: loglinear[k] for k in ("r2_mean", "r2_std", "r2_per_seed")},
            "image_poi": {k: image_poi[k] for k in ("r2_mean", "r2_std", "r2_per_seed")},
            "image_poi_light": {k: image_poi_light[k]
                                for k in ("r2_mean", "r2_std", "r2_per_seed")},
        },
        "light_alone_minus_image_poi": float(isotonic["r2_mean"] - image_poi["r2_mean"]),
        "oracle_light_gain": float(image_poi_light["r2_mean"] - image_poi["r2_mean"]),
        "residual_alignment": {
            "spearman_light_residual_vs_model_underprediction": float(rho),
            "p_value": float(p_value),
            "n_tiles": int(both.sum()),
        },
        "light_residual_by_decile": {
            "top_decile_mean_residual": float(np.nanmean(light_resid[top_decile])),
            "rest_mean_residual": float(np.nanmean(light_resid[rest])),
            "top_decile_share_of_squared_light_residual": float(
                np.nansum(light_resid[top_decile] ** 2)
                / np.nansum(light_resid[both] ** 2)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=REPORT_DIR / "urbanmoe_carbon_from_light.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    cities = {}
    for city in CITIES:
        logging.info("analysing %s", city)
        cities[city] = analyse_city(city)
        arms = cities[city]["arms"]
        logging.info("  light-alone isotonic R2 %.3f | image+POI R2 %.3f | +light oracle %.3f",
                     arms["light_isotonic"]["r2_mean"], arms["image_poi"]["r2_mean"],
                     arms["image_poi_light"]["r2_mean"])

    payload = {
        "question": "is the carbon target a monotone rendering of the nighttime-light target, "
                    "and is the residual the part the benchmark models fail on?",
        "warning": "the light label is a TARGET. These arms are target diagnostics, never "
                   "benchmark scores, and must never be reported as method results.",
        "protocol": {"seeds": list(SEEDS), "split": "60/20/20 random, identical across arms",
                     "scoring": "R2 in the ORIGINAL carbon space for every arm"},
        "cities": cities,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logging.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
