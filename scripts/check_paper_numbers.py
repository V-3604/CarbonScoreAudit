"""Check every number quoted in the paper against the reports in reports/.

Each number is compared at the precision the paper prints it. Exits with status 1 and lists the
mismatches if any number disagrees.

Run:
    python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import argparse
import json
import logging

from common import CITIES, REPORT_DIR, configure_logging


class Checker:
    def __init__(self) -> None:
        self.n_checks = 0
        self.failures: list[str] = []

    def equal(self, label: str, value: object, expected: object) -> None:
        self.n_checks += 1
        if value != expected:
            self.failures.append(f"{label}: paper says {expected}, report gives {value}")

    def rounded(self, label: str, value: float, expected: float, decimals: int) -> None:
        self.equal(label, round(value, decimals), expected)

    def percent(self, label: str, fraction: float, expected: float, decimals: int = 0) -> None:
        self.equal(label, round(100 * fraction, decimals), expected)

    def significant(self, label: str, value: float, expected: float, figures: int = 2) -> None:
        self.equal(label, float(f"{value:.{figures}g}"), expected)


def load(name: str) -> dict:
    path = REPORT_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the script that writes it first (see README).")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    configure_logging()
    light = load("carbon_from_light.json")["cities"]
    vintage = load("odiac_vintage.json")["cities"]
    sources = load("point_sources.json")
    screen = load("label_screen.json")
    sufficiency = load("sufficiency_test.json")["cities"]
    check = Checker()

    # Section 2: benchmark, provenance and scale.
    for city, n_tiles, r2, spearman in (("shanghai", 1558, 0.7858, 0.9697),
                                        ("beijing", 2310, 0.4403, 0.9959),
                                        ("newyork", 1091, 0.4560, 0.9683)):
        check.equal(f"{city} tiles", light[city]["n_tiles"], n_tiles)
        check.rounded(f"{city} pipeline carbon R2", light[city]["arms"]["image_poi"]["r2_mean"],
                      r2, 4)
        check.rounded(f"{city} Spearman of rebuild", vintage[city]["spearman_at_best_year"],
                      spearman, 4)
        check.equal(f"{city} match varies < 0.002 over years",
                    vintage[city]["r2_range_over_years"] < 0.002, True)
        check.equal(f"{city} match varies < 0.002 over versions",
                    vintage[city]["r2_range_over_versions"] < 0.002, True)
    check.equal("benchmark tiles", sum(light[c]["n_tiles"] for c in CITIES), 4959)
    check.rounded("New York label scale over Beijing",
                  sources["newyork_label_scale_over"]["beijing"], 9.0, 1)
    check.rounded("New York label scale over Shanghai",
                  sources["newyork_label_scale_over"]["shanghai"], 12.2, 1)

    # Table 1 and the nightlight paragraph.
    for city, pipeline, loglinear, isotonic, recovered, ratio_min, ratio_max, wins in (
            ("shanghai", 0.786, 0.424, 0.399, 54, 0.398, 0.719, 0),
            ("beijing", 0.440, 0.369, 0.350, 84, 0.756, 0.918, 0),
            ("newyork", 0.456, 0.507, 0.396, 111, 0.904, 1.378, 7)):
        arms = light[city]["arms"]
        check.rounded(f"Table 1 {city} image+POI", arms["image_poi"]["r2_mean"], pipeline, 3)
        check.rounded(f"Table 1 {city} log-linear", arms["light_loglinear"]["r2_mean"],
                      loglinear, 3)
        check.rounded(f"Table 1 {city} isotonic", arms["light_isotonic"]["r2_mean"], isotonic, 3)
        check.percent(f"Table 1 {city} fraction recovered", light[city]["fraction_recovered"],
                      recovered)
        check.rounded(f"Table 1 {city} per-seed ratio min", light[city]["per_seed_ratio_min"],
                      ratio_min, 3)
        check.rounded(f"Table 1 {city} per-seed ratio max", light[city]["per_seed_ratio_max"],
                      ratio_max, 3)
        check.equal(f"{city} seeds the light fit wins",
                    light[city]["seeds_light_beats_pipeline"], wins)
    check.significant("New York Wilcoxon p", light["newyork"]["wilcoxon_p"], 0.0645, 3)

    # Concentration: flagged cells against the WRI plant database.
    for city, fossil, renewable in (("beijing", 88, 0), ("shanghai", 64, 0), ("newyork", 59, 23)):
        cross = sources["cities"][city]["plant_cross_check"]
        check.percent(f"{city} flagged cells near a fossil plant",
                      cross["fossil"]["frac_flagged_cells_near_a_plant"], fossil)
        check.percent(f"{city} flagged cells near a renewable plant",
                      cross["renewable"]["frac_flagged_cells_near_a_plant"], renewable)

    # Table 2, the variance paragraph and the falsified prediction.
    for city, flagged, frac, error, variance, conc, full, screened, p in (
            ("shanghai", 17, 1.09, 6.4, 2.1, 7.1, 0.786, 0.805, 0.33),
            ("beijing", 13, 0.56, 64.4, 48.9, 114.4, 0.440, 0.771, 0.0035),
            ("newyork", 29, 2.66, 56.9, 39.7, 24.8, 0.456, 0.754, 0.0066)):
        row = screen["cities"][city]
        check.equal(f"Table 2 {city} flagged tiles", row["n_flagged"], flagged)
        check.equal(f"{city} flagged tiles agree with point_sources.py",
                    sources["cities"][city]["n_tiles_with_point_mass"], flagged)
        check.percent(f"Table 2 {city} % of tiles", row["flagged_frac"], frac, 2)
        check.percent(f"Table 2 {city} % of squared error", row["error_share"], error, 1)
        check.percent(f"Table 2 {city} % of variance", row["variance_share"], variance, 1)
        check.rounded(f"Table 2 {city} error concentration", row["error_concentration"], conc, 1)
        check.rounded(f"Table 2 {city} R2 full", row["r2_full"], full, 3)
        check.rounded(f"Table 2 {city} R2 removed", row["r2_screened"], screened, 3)
        check.significant(f"Table 2 {city} p", row["screened_minus_full"]["p"], p)
    check.percent("Beijing share of squared error, abstract",
                  screen["cities"]["beijing"]["error_share"], 64)
    check.rounded("Beijing change in reported score, introduction",
                  screen["cities"]["beijing"]["screened_minus_full"]["mean_delta"], 0.33, 2)
    check.equal("pre-registered prediction held", screen["prediction_held"], False)

    # Figure 1 panel sizes and the point-sources paragraph.
    for city, n_both, carried, sources_n, deficit in (("shanghai", 1514, 1, 10, 26.8),
                                                      ("beijing", 2163, 7, 7, None),
                                                      ("newyork", 1063, 8, 15, 14.2)):
        city_sources = sources["cities"][city]
        carrying = city_sources["label_carrying"]
        check.equal(f"Figure 1 {city} n", city_sources["n_tiles_both_positive"], n_both)
        check.equal(f"{city} detected sources", carrying["n_source_tiles"], sources_n)
        check.equal(f"{city} sources the label carries", carrying["n_carried"], carried)
        if deficit is None:
            check.equal(f"{city} label total is not lower", carrying["deficit_frac"] <= 0, True)
        else:
            check.percent(f"{city} label total below the rebuild", carrying["deficit_frac"],
                          deficit, 1)

    # Sufficiency test.
    for city, delta, p, before, after in (("beijing", 0.191, 0.0084, 0.440, 0.631),
                                          ("newyork", 0.075, 0.0080, None, None),
                                          ("shanghai", -0.006, 0.056, None, None)):
        row = sufficiency[city]
        check.rounded(f"sufficiency {city} delta", row["with_minus_default"]["mean_delta"],
                      delta, 3)
        check.significant(f"sufficiency {city} p", row["with_minus_default"]["p"], p)
        if before is not None:
            check.rounded(f"sufficiency {city} R2 before", row["r2_default"], before, 3)
            check.rounded(f"sufficiency {city} R2 after", row["r2_with_point_features"], after, 3)

    # Appendix: year and version.
    for city, year, margin in (("shanghai", 2016, 0.00008), ("beijing", 2022, 0.00004),
                               ("newyork", 2020, 0.00003)):
        check.equal(f"{city} nominal best year", vintage[city]["best_year"], year)
        check.rounded(f"{city} margin over runner-up", vintage[city]["margin_over_runner_up"],
                      margin, 5)

    if check.failures:
        for failure in check.failures:
            logging.error(failure)
        logging.error("%d of %d checks failed", len(check.failures), check.n_checks)
        return 1
    logging.info("all %d numbers in the paper match the reports", check.n_checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
