"""How much of the carbon score the nightlight target reaches on its own (Table 1).

ODIAC spreads its non-point emissions by nighttime lights, and nighttime light is one of the
benchmark's three targets. Carbon is fitted from the light target alone in two monotone ways and
compared with the image+POI pipeline on the same ten 60/20/20 splits:

  light_loglinear   ridge from log1p(light) to log1p(carbon)
  light_isotonic    isotonic regression from light to carbon
  image_poi         the image+POI pipeline

The light target is never a model input in practice, so these arms describe the label rather
than any model. The recovered fraction is the log-linear score over the pipeline score, and the
per-seed ratio compares the two arms within each split.

Output: reports/carbon_from_light.json

Run:
    python scripts/carbon_from_light.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from scipy.stats import wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

from common import CARBON, CITIES, LIGHT, SEEDS, configure_logging, load_bundle, write_report
from probe import Split, ridge_probe, seed_splits


def summary(r2: np.ndarray) -> dict[str, object]:
    return {"r2_mean": float(r2.mean()), "r2_std": float(r2.std(ddof=1)),
            "r2_per_seed": [float(v) for v in r2]}


def isotonic_r2(light: np.ndarray, carbon: np.ndarray, splits: list[Split]) -> np.ndarray:
    scores = []
    for train, test in splits:
        model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        model.fit(light[train], carbon[train])
        scores.append(r2_score(carbon[test], model.predict(light[test])))
    return np.asarray(scores, dtype=np.float64)


def analyse_city(city: str) -> dict[str, object]:
    bundle = load_bundle(city)
    carbon, light = bundle.y[:, CARBON], bundle.y[:, LIGHT]
    splits = seed_splits(len(carbon))
    features, widths = bundle.image_poi()

    loglinear = ridge_probe(np.log1p(light)[:, None], carbon, splits).r2
    isotonic = isotonic_r2(light, carbon, splits)
    pipeline = ridge_probe(features, carbon, splits, block_widths=widths).r2
    ratio = loglinear / pipeline
    logging.info("%-9s light log-linear %.3f | isotonic %.3f | image+POI %.3f | recovered %.0f%%",
                 city, loglinear.mean(), isotonic.mean(), pipeline.mean(),
                 100 * loglinear.mean() / pipeline.mean())
    return {
        "n_tiles": int(len(carbon)),
        "arms": {"light_loglinear": summary(loglinear), "light_isotonic": summary(isotonic),
                 "image_poi": summary(pipeline)},
        "fraction_recovered": float(loglinear.mean() / pipeline.mean()),
        "per_seed_ratio_min": float(ratio.min()),
        "per_seed_ratio_max": float(ratio.max()),
        "seeds_light_beats_pipeline": int((loglinear > pipeline).sum()),
        "wilcoxon_p": float(wilcoxon(loglinear, pipeline).pvalue),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    configure_logging()
    report = {
        "seeds": list(SEEDS),
        "split": "60/20/20, identical across arms",
        "scoring": "R2 on raw carbon values",
        "cities": {city: analyse_city(city) for city in CITIES},
    }
    write_report("carbon_from_light.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
