"""Check how far the three benchmark targets are functions of one another.

Carbon inventories at this resolution are often downscaled from population and land use, in which
case predicting carbon from imagery partly recovers the inventory's own inputs. This measures the
coupling between the released targets before anything is built on them.

Reports, per city and pooled:
  - row counts, column names and value ranges
  - Pearson and Spearman between every target pair
  - R2 of each target predicted from the others by ordinary least squares

Input : data/raw/urbanmoe/
Output: artifacts/reports/urbanmoe_label_audit.json

Run:
    python scripts/urbanmoe_label_audit.py
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/raw/urbanmoe"
REPORT_PATH = REPO_ROOT / "artifacts/reports/urbanmoe_label_audit.json"

CITY_CSVS = {
    "shanghai": DATA_DIR / "shanghai/shanghai_merged_normalized.csv",
    "beijing": DATA_DIR / "beijing/Beijing_merged_normalized.csv",
    "newyork": DATA_DIR / "NewWork/NewYork_merged_normalized.csv",
}
TARGETS = ["c_value", "pop_value", "light_value"]
SHORT = {"c_value": "carbon", "pop_value": "population", "light_value": "light"}

# Above this, the target pair is collinear enough that a reviewer will read one as a proxy for
# the other and the headline needs an explicit paragraph rather than a footnote.
COLLINEARITY_FLAG = 0.80


@dataclass
class CityAudit:
    city: str
    n_rows: int
    columns: list[str]
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    pearson: dict[str, float] = field(default_factory=dict)
    spearman: dict[str, float] = field(default_factory=dict)
    r2_from_others: dict[str, float] = field(default_factory=dict)


def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R^2 of an intercept-bearing least-squares fit of y on X. In-sample by design.

    In-sample is the right choice here: the question is how much of the target is linearly
    determined by the others, not how well it generalises.
    """
    design = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coeffs
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0.0:
        raise ValueError("target has zero variance; the CSV column is constant")
    return float(1.0 - (residual**2).sum() / total)


def audit_city(city: str, csv_path: Path) -> CityAudit:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. The UrbanMoE release must be extracted to "
            f"{DATA_DIR} first."
        )
    frame = pd.read_csv(csv_path)
    missing = [t for t in TARGETS if t not in frame.columns]
    if missing:
        raise KeyError(f"{csv_path.name} is missing target columns {missing}; found {list(frame.columns)}")

    audit = CityAudit(city=city, n_rows=len(frame), columns=list(frame.columns))
    for target in TARGETS:
        values = frame[target].to_numpy(dtype=np.float64)
        audit.ranges[SHORT[target]] = (float(values.min()), float(values.max()))

    for a, b in combinations(TARGETS, 2):
        key = f"{SHORT[a]}~{SHORT[b]}"
        x, y = frame[a].to_numpy(dtype=np.float64), frame[b].to_numpy(dtype=np.float64)
        audit.pearson[key] = round(float(stats.pearsonr(x, y).statistic), 4)
        audit.spearman[key] = round(float(stats.spearmanr(x, y).statistic), 4)

    for target in TARGETS:
        others = [t for t in TARGETS if t != target]
        X = frame[others].to_numpy(dtype=np.float64)
        y = frame[target].to_numpy(dtype=np.float64)
        audit.r2_from_others[SHORT[target]] = round(ols_r2(X, y), 4)
        for other in others:
            single = frame[[other]].to_numpy(dtype=np.float64)
            audit.r2_from_others[f"{SHORT[target]}<-{SHORT[other]}"] = round(ols_r2(single, y), 4)

    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPORT_PATH, help="where to write the JSON report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audits = {city: audit_city(city, path) for city, path in CITY_CSVS.items()}

    for city, audit in audits.items():
        logging.info("\n=== %s: %d rows, columns %s", city, audit.n_rows, audit.columns)
        for key, (lo, hi) in audit.ranges.items():
            logging.info("    %-11s range %.4f .. %.4f", key, lo, hi)
        for key in audit.pearson:
            logging.info("    %-22s pearson %+.3f   spearman %+.3f",
                         key, audit.pearson[key], audit.spearman[key])
        logging.info("    carbon from population alone   R2 = %.4f",
                     audit.r2_from_others["carbon<-population"])
        logging.info("    carbon from population + light R2 = %.4f",
                     audit.r2_from_others["carbon"])

    flagged = {
        city: {k: v for k, v in a.pearson.items() if abs(v) >= COLLINEARITY_FLAG}
        for city, a in audits.items()
    }
    verdict = {
        "collinearity_flag": COLLINEARITY_FLAG,
        "flagged_pairs": {c: p for c, p in flagged.items() if p},
        "carbon_from_population_r2": {c: a.r2_from_others["carbon<-population"] for c, a in audits.items()},
    }
    worst = max(verdict["carbon_from_population_r2"].values())
    logging.info(
        "\ncarbon is at most %.1f%% linearly explained by population alone (worst city). "
        "%s", 100 * worst,
        "REPORT AND DISCUSS THIS EXPLICITLY -- a reviewer will ask." if worst >= 0.5
        else "Low enough that carbon is not a population proxy; state it in one sentence.",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": "are the UrbanMoE targets independent constructs, or is carbon a population proxy?",
        "verdict": verdict,
        "per_city": {c: a.__dict__ for c, a in audits.items()},
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logging.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
