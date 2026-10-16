"""Ridge probe on frozen features, scored on the 60/20/20 protocol used for every result.

Each split standardises features on its training rows, fits RidgeCV and scores R2 on its test
rows. Carbon is fitted on log1p and the prediction is mapped back with expm1, so R2 is always
computed on raw values. When several feature blocks are concatenated, each block is divided by
the square root of its width after standardisation so that a wide block (1152 image dimensions)
does not take the ridge penalty away from a narrow one (38 POI columns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from common import SEEDS

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

Split = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ProbeResult:
    """R2 per split, and the (test rows, prediction) pair of every split."""

    r2: np.ndarray
    predictions: list[tuple[np.ndarray, np.ndarray]]


def split_60_20_20(n: int, seed: int) -> Split:
    """(train, test) row indices. The 20% validation rows are held out but unused, since RidgeCV
    selects its penalty internally."""
    order = np.random.default_rng(seed).permutation(n)
    n_test = int(round(0.2 * n))
    n_val = int(round(0.2 * n))
    return order[n_test + n_val:], order[:n_test]


def seed_splits(n: int, seeds: tuple[int, ...] = SEEDS) -> list[Split]:
    return [split_60_20_20(n, seed) for seed in seeds]


def block_scale(x: np.ndarray, widths: tuple[int, ...]) -> np.ndarray:
    out = x.copy()
    start = 0
    for width in widths:
        out[:, start:start + width] /= np.sqrt(width)
        start += width
    if start != x.shape[1]:
        raise ValueError(f"block widths sum to {start} but the matrix has {x.shape[1]} columns")
    return out


def ridge_probe(features: np.ndarray, y: np.ndarray, splits: list[Split],
                block_widths: tuple[int, ...] | None = None) -> ProbeResult:
    """Fit on log1p(y) and score R2 on raw y for every split."""
    if np.any(y < 0):
        raise ValueError("the log1p target needs non-negative values")
    y_log = np.log1p(y)
    scores, predictions = [], []
    for train, test in splits:
        scaler = StandardScaler().fit(features[train])
        x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])
        if block_widths:
            x_train, x_test = block_scale(x_train, block_widths), block_scale(x_test, block_widths)
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(x_train, y_log[train])
        pred = np.clip(np.expm1(model.predict(x_test)), 0.0, None)
        scores.append(r2_score(y[test], pred))
        predictions.append((test, pred))
    return ProbeResult(r2=np.asarray(scores, dtype=np.float64), predictions=predictions)
