"""Frozen encoder plus linear probe on the three benchmark targets.

This is the pipeline every number in the paper is measured against. One frozen encoder (SigLIP),
a per-target ridge probe, StandardScaler fit on the training split only, ten random 60/20/20
seeds reported as mean and standard deviation.

R2 is computed in the original target space even when a model fits a transformed target, so
scores stay comparable across the transformed and untransformed arms.

Targets are carbon (c_value), population (pop_value) and nighttime light (light_value) for
Shanghai, Beijing and New York.

Input : artifacts/features/urbanmoe_{city}.npz
Output: artifacts/reports/urbanmoe_probe.json

Run:
    python scripts/urbanmoe_probe.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/raw/urbanmoe"
EMBED_DIR = REPO_ROOT / "artifacts/embeddings"
REPORT_DIR = REPO_ROOT / "artifacts/reports"
OUT_JSON = REPORT_DIR / "urbanmoe_probe_results.json"

CITIES: dict[str, str] = {
    "shanghai": "shanghai_merged_normalized.csv",
    "beijing": "Beijing_merged_normalized.csv",
    "newyork": "NewYork_merged_normalized.csv",
}
CITY_FOLDER = {"shanghai": "shanghai", "beijing": "beijing", "newyork": "NewWork"}
TARGETS = ("c_value", "pop_value", "light_value")
TARGET_LABEL = {"c_value": "carbon", "pop_value": "population", "light_value": "light"}
SEEDS = tuple(range(41, 51))
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

# Verified Table 2 (R2), per city, per method, ordered (carbon, population, light).
# Third city: paper = "Washington DC", released data = "New York" (same set).
TABLE2_R2: dict[str, dict[str, tuple[float, float, float]]] = {
    "shanghai": {"UrbanMoE": (0.7837, 0.7872, 0.8001),
                 "RemoteCLIP-MT": (0.7450, 0.6941, 0.7809),
                 "UrbanCLIP-MT": (0.6835, 0.6595, 0.7559)},
    "beijing": {"UrbanMoE": (0.7798, 0.8625, 0.8642),
                "RemoteCLIP-MT": (0.5896, 0.8557, 0.8424),
                "UrbanCLIP-MT": (0.6553, 0.8599, 0.8156)},
    "newyork": {"UrbanMoE": (0.6971, 0.8775, 0.8687),
                "RemoteCLIP-MT": (0.4610, 0.6923, 0.8607),
                "UrbanCLIP-MT": (0.6830, 0.7168, 0.8502)},
}
TARGET_INDEX = {"c_value": 0, "pop_value": 1, "light_value": 2}


def load_city(city: str, encoder: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Frozen features + label frame aligned to the embedding row order (by filename)."""
    emb_path = EMBED_DIR / f"urbanmoe_{encoder}_{city}.npy"
    files_path = EMBED_DIR / f"urbanmoe_{encoder}_{city}_files.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found. Run urbanmoe_embed.py first (GPU pass over the tiles).")
    features = np.load(emb_path).astype(np.float64)
    files = np.load(files_path, allow_pickle=True).astype(str)
    csv = pd.read_csv(DATA_DIR / CITY_FOLDER[city] / CITIES[city]).set_index("filename")
    labels = csv.loc[files].reset_index()   # align labels to embedding order
    return features, labels


def split_60_20_20(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train 60%, test 20%) index arrays; the 20% val is unused by a linear probe."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(0.2 * n))
    n_val = int(round(0.2 * n))
    test = idx[:n_test]
    train = idx[n_test + n_val:]
    return train, test


def _block_scale(x: np.ndarray, block_sizes: tuple[int, ...]) -> np.ndarray:
    """Divide each block's columns by sqrt(block width) so blocks share the ridge penalty evenly.

    After StandardScaler every column has unit variance, so a d-column block contributes d to
    the total feature variance and therefore soaks up the shared L2 penalty in proportion to d.
    Concatenating 1152 image dims with 38 POI dims lets the image block regularise the POI block
    away: on Beijing carbon, POI alone scored R2 0.388 against image's 0.277, yet image+POI came
    out at 0.289 -- worse than the better of its parts, which is a probe artifact and not a
    finding. Dividing by sqrt(d) equalises each block's variance contribution (standard
    multiblock practice). Single-block callers are unaffected: a uniform rescaling is absorbed
    by RidgeCV's alpha search.
    """
    out = x.copy()
    start = 0
    for width in block_sizes:
        out[:, start:start + width] /= np.sqrt(width)
        start += width
    if start != x.shape[1]:
        raise ValueError(f"block_sizes sum to {start} but the matrix has {x.shape[1]} columns")
    return out


def probe_target(features: np.ndarray, y: np.ndarray,
                 block_sizes: tuple[int, ...] | None = None,
                 target_transform: str | None = None,
                 seeds: tuple[int, ...] = SEEDS,
                 return_per_seed: bool = False,
                 splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
                 return_predictions: bool = False) -> dict[str, float]:
    """Ridge probe over SEEDS; returns mean/std of R2, RMSE, MAE on the held-out test split.

    `block_sizes` (widths of the concatenated modality blocks) enables block scaling; see
    `_block_scale`.

    `target_transform="log1p"` fits the probe on log1p(y) and then inverts the prediction with
    expm1 before scoring. **Every reported metric is computed in the ORIGINAL target space**, so
    the transform is a modelling choice and not a metric change, and the numbers stay directly
    comparable to Table 2 and to the untransformed arms. Motivation is measured, not assumed:
    `urbanmoe_label_audit.py` finds carbon skew 15.4 (Beijing) and 11.2 (New York) against 4.2
    (Shanghai), with medians of 0.69/3.15 against maxima of 224/217, and those are exactly the two
    cells where the raw-target probe collapses. Inverted predictions are clipped at 0 because the
    targets are non-negative counts/intensities; `r2_in_log_space` is reported for diagnosis only
    and must never be compared against Table 2.

    `seeds` defaults to the 10-seed protocol constant, `return_per_seed` off, and `splits` None,
    so every call written before 2026-08-04 reproduces its number exactly. The overrides exist for
    the benchmark-variance study (`urbanmoe_seed_variance.py`), which needs the raw per-split
    scores and more than 10 draws, and for the spatial-block split (`urbanmoe_moe.py --split
    spatial`), which supplies its own (train, test) index pairs so neighbouring tiles cannot
    straddle the split.
    """
    if target_transform not in (None, "log1p"):
        raise ValueError(f"unknown target_transform {target_transform!r}; expected None or 'log1p'")
    if target_transform == "log1p" and np.any(y < 0):
        raise ValueError("log1p target transform requires non-negative targets; got negatives")

    y_fit = np.log1p(y) if target_transform == "log1p" else y
    r2s, rmses, maes, r2s_transformed = [], [], [], []
    fold_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    folds = splits if splits is not None else [split_60_20_20(len(y), s) for s in seeds]
    for train, test in folds:
        scaler = StandardScaler().fit(features[train])
        x_tr, x_te = scaler.transform(features[train]), scaler.transform(features[test])
        if block_sizes:
            x_tr, x_te = _block_scale(x_tr, block_sizes), _block_scale(x_te, block_sizes)
        probe = RidgeCV(alphas=RIDGE_ALPHAS).fit(x_tr, y_fit[train])
        pred = probe.predict(x_te)
        if target_transform == "log1p":
            r2s_transformed.append(r2_score(y_fit[test], pred))
            pred = np.clip(np.expm1(pred), 0.0, None)
        r2s.append(r2_score(y[test], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y[test], pred))))
        maes.append(float(mean_absolute_error(y[test], pred)))
        if return_predictions:
            fold_predictions.append((test, np.asarray(pred)))
    stats = {
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s, ddof=1)),
        "rmse_mean": float(np.mean(rmses)), "mae_mean": float(np.mean(maes)),
        "n_seeds": len(folds),
    }
    if target_transform == "log1p":
        stats["r2_in_log_space"] = float(np.mean(r2s_transformed))
    if return_per_seed:
        stats["r2_per_seed"] = [float(v) for v in r2s]
        stats["seeds"] = list(seeds) if splits is None else list(range(len(folds)))
    if return_predictions:
        stats["fold_predictions"] = fold_predictions
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Frozen encoder + linear probe on the UrbanMoE benchmark.")
    parser.add_argument("--encoder", default="siglip")
    parser.add_argument("--cities", nargs="+", default=list(CITIES))
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {"meta": {"encoder": args.encoder, "probe": "ridge, single-task, per-target",
                              "seeds": list(SEEDS), "split": "60/20/20 random (val unused)",
                              "note": "our simple frozen+linear (single-task, imagery-only) vs their "
                                      "the benchmark's published multi-task multi-modal results"},
                     "cities": {}}
    win = tie = lose = 0
    for city in args.cities:
        features, labels = load_city(city, args.encoder)
        logging.info("=== %s (n=%d, dim=%d) — frozen %s + ridge probe ===",
                     city, len(features), features.shape[1], args.encoder)
        results["cities"][city] = {}
        for target in TARGETS:
            y = labels[target].to_numpy(dtype=np.float64)
            stats = probe_target(features, y)
            moe = TABLE2_R2[city]["UrbanMoE"][TARGET_INDEX[target]]
            remoteclip = TABLE2_R2[city]["RemoteCLIP-MT"][TARGET_INDEX[target]]
            delta = stats["r2_mean"] - moe
            verdict = "beats/ties MoE" if delta >= -0.02 else "below MoE"
            win += delta > 0.02
            tie += abs(delta) <= 0.02
            lose += delta < -0.02
            stats.update({"urbanmoe_r2": moe, "remoteclip_mt_r2": remoteclip,
                          "delta_vs_moe": round(delta, 4), "verdict": verdict})
            results["cities"][city][target] = stats
            logging.info("  %-11s ours R2=%.3f±%.3f | MoE %.3f | RemoteCLIP-MT %.3f | Δvs MoE %+.3f  %s",
                         TARGET_LABEL[target], stats["r2_mean"], stats["r2_std"], moe, remoteclip,
                         delta, verdict)

    results["summary"] = {"frozen_linear_vs_moe": {"beats": win, "ties": tie, "below": lose,
                          "n_cells": win + tie + lose}}
    logging.info("SUMMARY: frozen SigLIP + linear probe vs UrbanMoE across %d (city×target) cells — "
                 "beats %d, ties %d, below %d (tie = |ΔR²|≤0.02).", win + tie + lose, win, tie, lose)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logging.info("Wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
