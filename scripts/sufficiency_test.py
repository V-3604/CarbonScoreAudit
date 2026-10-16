"""Does adding the label's own point-source values as features close the carbon gap?

Scores the image+POI pipeline on carbon with and without the three point-source columns written
by point_features.py, on the same ten 60/20/20 splits, and tests the per-seed difference with a
paired t-test. This is a sufficiency test, not a performance result: the added columns are
inputs from which the label was built.

Output: reports/sufficiency_test.json

Run:
    python scripts/sufficiency_test.py
"""

from __future__ import annotations

import argparse
import logging

from common import CARBON, CITIES, SEEDS, configure_logging, load_bundle, paired_t, write_report
from point_features import VARIANT
from probe import ridge_probe, seed_splits


def score_city(city: str) -> dict[str, object]:
    scores = {}
    for name, variant in (("default", ""), ("with_point_features", VARIANT)):
        bundle = load_bundle(city, variant)
        features, widths = bundle.image_poi()
        carbon = bundle.y[:, CARBON]
        scores[name] = ridge_probe(features, carbon, seed_splits(len(carbon)),
                                   block_widths=widths).r2
    result = {
        "r2_default": float(scores["default"].mean()),
        "r2_with_point_features": float(scores["with_point_features"].mean()),
        "with_minus_default": paired_t(scores["with_point_features"], scores["default"]),
    }
    logging.info("%-9s carbon R2 %.3f -> %.3f (delta %+.3f, p=%.2g)", city,
                 result["r2_default"], result["r2_with_point_features"],
                 result["with_minus_default"]["mean_delta"], result["with_minus_default"]["p"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    configure_logging()
    report = {
        "seeds": list(SEEDS),
        "target": "carbon",
        "cities": {city: score_city(city) for city in CITIES},
    }
    write_report("sufficiency_test.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
