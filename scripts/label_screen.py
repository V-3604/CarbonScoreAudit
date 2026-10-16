"""Re-score the image+POI pipeline without the tiles that hold injected point sources (Table 2).

A tile is flagged when it overlaps a cell flagged by the rule in point_sources.py (K = 20,
ODIAC2024, 2015). Nothing is retrained and the model never sees the point term: each of the ten
splits is scored twice, once on its full test set and once with the flagged tiles removed. Since
removing tiles removes variance as well as error, the flagged tiles' share of each is reported.

Before the first run we predicted that the flagged tiles would carry at least five times their
tile share of squared error in Beijing and New York, and less than five times in Shanghai.
Shanghai came in above five times, so the prediction was wrong; the report records the outcome.

Output: reports/label_screen.json

Run:
    python scripts/label_screen.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from sklearn.metrics import r2_score

from common import CARBON, CITIES, SEEDS, configure_logging, load_bundle, paired_t, write_report
from odiac import load_odiac_city
from point_sources import detect_city, point_mass_per_tile
from probe import ridge_probe, seed_splits

PREDICTED_MULTIPLE = 5.0


def flagged_tiles(city: str) -> np.ndarray:
    data = load_odiac_city(city)
    values, detection = detect_city(data)
    return point_mass_per_tile(values, detection, data) > 0


def screen_city(city: str) -> dict[str, object]:
    bundle = load_bundle(city)
    carbon = bundle.y[:, CARBON]
    features, widths = bundle.image_poi()
    flagged = flagged_tiles(city)
    probe = ridge_probe(features, carbon, seed_splits(len(carbon)), block_widths=widths)

    r2_full, r2_screened, error_share, variance_share, tile_share = [], [], [], [], []
    for test, pred in probe.predictions:
        y_test, is_flagged = carbon[test], flagged[test]
        squared_error = (y_test - pred) ** 2
        squared_deviation = (y_test - y_test.mean()) ** 2
        r2_full.append(r2_score(y_test, pred))
        r2_screened.append(r2_score(y_test[~is_flagged], pred[~is_flagged]))
        error_share.append(squared_error[is_flagged].sum() / squared_error.sum())
        variance_share.append(squared_deviation[is_flagged].sum() / squared_deviation.sum())
        tile_share.append(is_flagged.mean())

    r2_full, r2_screened = np.asarray(r2_full), np.asarray(r2_screened)
    concentration = float(np.mean(error_share) / np.mean(tile_share))
    result = {
        "n_tiles": int(len(carbon)),
        "n_flagged": int(flagged.sum()),
        "flagged_frac": float(flagged.mean()),
        "error_share": float(np.mean(error_share)),
        "variance_share": float(np.mean(variance_share)),
        "error_concentration": concentration,
        "r2_full": float(r2_full.mean()),
        "r2_screened": float(r2_screened.mean()),
        "r2_full_per_seed": [float(v) for v in r2_full],
        "r2_screened_per_seed": [float(v) for v in r2_screened],
        "screened_minus_full": paired_t(r2_screened, r2_full),
    }
    logging.info("%-9s %d/%d flagged, %.1f%% of error, %.1fx, R2 %.3f -> %.3f (p=%.2g)",
                 city, result["n_flagged"], result["n_tiles"], 100 * result["error_share"],
                 concentration, result["r2_full"], result["r2_screened"],
                 result["screened_minus_full"]["p"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    configure_logging()
    cities = {city: screen_city(city) for city in CITIES}
    multiple = {city: cities[city]["error_concentration"] for city in CITIES}
    report = {
        "seeds": list(SEEDS),
        "flag": "tile overlaps a point-source cell (K = 20, ODIAC2024, 2015)",
        "prediction": f"error concentration >= {PREDICTED_MULTIPLE} in Beijing and New York, "
                      f"< {PREDICTED_MULTIPLE} in Shanghai",
        "prediction_held": bool(multiple["beijing"] >= PREDICTED_MULTIPLE
                                and multiple["newyork"] >= PREDICTED_MULTIPLE
                                and multiple["shanghai"] < PREDICTED_MULTIPLE),
        "cities": cities,
    }
    write_report("label_screen.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
