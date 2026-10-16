# What Does a Carbon Score Measure?

Code for *What Does a Carbon Score Measure? A Case Study of Three Cities in an Urban Emissions
Benchmark*, accepted at Tackling Climate Change with Machine Learning, workshop at NeurIPS 2026.

Several recent urban prediction models train on ODIAC carbon as a target. ODIAC builds its urban
spatial pattern from two components: nighttime lights, which is itself a target in these
benchmarks, and a power-plant database with a 2007 base year. This repository contains the code
that measures how much of a carbon score on a released three-city benchmark each component
explains, and the detection rule that flags the affected tiles from the raster alone.

## Requirements

Python 3.11.

```
pip install -r requirements.txt
```

CPU is sufficient for every analysis step. The embedding step (`urbanmoe_embed.py`) is much faster
on a GPU but does not require one.

## Data

No third-party data is redistributed here. Each source is public and must be obtained from its
provider:

| Source | Used for | Licence |
|---|---|---|
| The benchmark release (4,959 tiles, Shanghai / Beijing / New York, with `c_value`, `pop_value`, `light_value`) | tiles and targets | per its authors |
| [ODIAC](https://db.cger.nies.go.jp/dataset/ODIAC/) 1 km monthly rasters, fetched through the [US Greenhouse Gas Center](https://earth.gov) raster API | rebuilding the carbon label | CC BY 4.0 |
| [OpenStreetMap](https://download.geofabrik.de/) regional extracts (`.osm.pbf`) | POI and land-use blocks | ODbL |
| [WRI Global Power Plant Database v1.3.0](https://datasets.wri.org/dataset/globalpowerplantdatabase) | plant cross-check and placebo | CC BY 4.0 |

Expected layout:

```
data/raw/urbanmoe/                          benchmark tiles and label CSVs
data/raw/osm_extracts/*.osm.pbf             OSM regional extracts
data/external/global_power_plant_database.csv
data/external/odiac/crops/                  written by the vintage sweep
```

## Reproducing the paper

Run in order. Every step writes a JSON report to `artifacts/reports/`; the reports behind the
published numbers are included in `artifacts/` so the tables can be checked without re-running
anything.

**Features**

```
python scripts/urbanmoe_embed.py                          # frozen SigLIP embeddings
python scripts/urbanmoe_poi_density.py --dry-run          # tile bounding-box index
python scripts/urbanmoe_poi_density_local.py              # POI counts, all tiles
python scripts/urbanmoe_osm_layers.py                     # emitter / industry / built layers
python scripts/urbanmoe_build_features.py                 # aligned per-city bundles
```

**Analysis**

| Command | Produces |
|---|---|
| `python scripts/urbanmoe_probe.py` | image+POI pipeline scores (carbon 0.786 / 0.440 / 0.456) |
| `python scripts/urbanmoe_probe_poi.py` | block ablation: image only, POI only, image+POI (image+text+POI when text embeddings are present) |
| `python scripts/urbanmoe_label_audit.py` | coupling between the three targets |
| `python scripts/urbanmoe_odiac_vintage.py` | ODIAC year and version sweep (appendix) |
| `python scripts/urbanmoe_carbon_from_light.py` | **Table 1**, carbon recoverable from the light target (54% / 84% / 111%) |
| `python scripts/urbanmoe_power_plants.py` | worst-tile proximity to real plants, with renewable placebo |
| `python scripts/urbanmoe_odiac_pointsources.py --year 2015` | point-source detection rule and plant cross-check |
| `python scripts/urbanmoe_label_screen.py` | **Table 2**, error and variance shares, screened scores |
| `python scripts/urbanmoe_plant_features.py` | sufficiency test |
| `python scripts/make_odiac_figure.py --year 2015` | **Figure 1** |

Scores use a 60/20/20 split over ten seeds, with R2 computed in the original target space. The
plant-proximity diagnostics in `urbanmoe_power_plants.py` and `urbanmoe_odiac_pointsources.py` use
a separate 5-fold protocol; the two are never mixed, and no score from the 5-fold arm appears in
the paper's tables.

## Layout

```
scripts/     analysis pipeline
  common.py      shared paths and feature-bundle loading
  plotting.py    figure styling and output
artifacts/   result JSONs behind the reported numbers, plus the tile bbox index
figures/     written by make_odiac_figure.py
```

## Notes on interpretation

Two framings in the paper are load-bearing and are repeated in the code:

The tile removal in `urbanmoe_label_screen.py` is a reporting intervention, not a feature. The
model is never shown the point term and nothing is retrained; only which tiles count toward the
reported score changes. Removing tiles removes variance as well as error, so both shares are
reported, and the score rises only where the error share exceeds the variance share.

The feature test in `urbanmoe_plant_features.py` is a sufficiency test, not a performance result.
The columns that close the gap are inputs to the label's construction, which no deployed model
would have.

## Citation

```bibtex
@inproceedings{gowda2026carbon,
  title     = {What Does a Carbon Score Measure? A Case Study of Three Cities
               in an Urban Emissions Benchmark},
  author    = {Gowda, Varshith},
  booktitle = {Tackling Climate Change with Machine Learning: workshop at NeurIPS 2026},
  year      = {2026}
}
```

## Licence

Code is released under the MIT Licence. The data sources listed above carry their own licences.
