"""Extract the OSM layers corresponding to what ODIAC builds carbon from.

ODIAC places large point sources from a power-plant database and spreads the remainder using
nighttime lights. A POI vocabulary of amenity, shop and public_transport therefore has no
category corresponding to the point-source term. This script adds layers that do.

Feature groups, each tied to a documented property of the target:
  emitters    power=plant|generator|substation, man_made=works|chimney|storage_tank|gasometer,
              aeroway=aerodrome, corresponding to ODIAC's point-source layer
  industry    landuse=industrial and related area fractions
  built       building footprint counts and area fractions

Area columns are tile-area fractions so they are comparable across tiles and cities.

Input : data/raw/osm_extracts/*.osm.pbf
        artifacts/reports/urbanmoe_tile_bboxes.json
Output: artifacts/reports/urbanmoe_osm_layers.json

Run:
    python scripts/urbanmoe_osm_layers.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from urbanmoe_poi_density_local import (
    CITY_EXTRACT, EXTRACT_DIR, REPORT_DIR, EMBED_DIR, TileBox, assign_tiles, load_tiles,
)

OUT_JSON = REPORT_DIR / "urbanmoe_osm_layers.json"
VOCAB_JSON = EMBED_DIR / "urbanmoe_osm_vocab.json"

EMITTER_TAGS: dict[str, tuple[str, ...]] = {
    "power": ("plant", "generator", "substation"),
    "man_made": ("works", "chimney", "storage_tank", "gasometer"),
    "aeroway": ("aerodrome",),
}
LANDUSE_VALUES = ("industrial", "commercial", "retail", "residential", "construction",
                  "quarry", "military", "railway")
ROAD_VALUES = ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service")


@dataclass
class CityHarvest:
    """Everything pulled out of one extract for one city, in tile-assignable form."""

    emitter_points: list[tuple[float, float]] = field(default_factory=list)
    emitter_kinds: list[str] = field(default_factory=list)
    # Buildings are small relative to a 2 km tile, so charging them to their centroid's tile is
    # accurate enough. Landuse polygons routinely span several tiles, so they keep their geometry
    # and are intersected against the tile grid instead.
    building_points: list[tuple[float, float]] = field(default_factory=list)
    building_m2: list[float] = field(default_factory=list)
    landuse_polys: list[tuple[str, list[tuple[float, float]]]] = field(default_factory=list)
    road_points: list[tuple[float, float]] = field(default_factory=list)
    road_kinds: list[str] = field(default_factory=list)
    road_m: list[float] = field(default_factory=list)


def feature_names() -> list[str]:
    names = [f"emitter:{key}={value}" for key, values in EMITTER_TAGS.items() for value in values]
    names.append("emitter:total")
    names += [f"landuse_frac:{value}" for value in LANDUSE_VALUES]
    names += ["building:count", "building:total_m2", "building:mean_m2"]
    names += [f"road_m:{value}" for value in ROAD_VALUES]
    names.append("road_m:total")
    return names


def _projector(tiles: list[TileBox]):
    """Local equirectangular lon/lat -> metres about the city centroid."""
    lat0 = float(np.mean([(t.south + t.north) / 2 for t in tiles]))
    lon0 = float(np.mean([(t.west + t.east) / 2 for t in tiles]))
    scale_x = 111_320.0 * np.cos(np.radians(lat0))

    def project(coords: list[tuple[float, float]]) -> np.ndarray:
        arr = np.asarray(coords, dtype=np.float64)
        return np.column_stack([(arr[:, 0] - lon0) * scale_x, (arr[:, 1] - lat0) * 110_540.0])

    return project


def _polygon_area_m2(coords: list[tuple[float, float]], project) -> float:
    xy = project(coords)
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def harvest(path: Path, bounds: tuple[float, float, float, float], project,
            index_type: str) -> CityHarvest:
    """One pass over the extract, keeping only objects inside the city envelope."""
    import osmium

    west, south, east, north = bounds
    out = CityHarvest()
    n_seen = 0
    started = time.time()

    def inside(lon: float, lat: float) -> bool:
        return west <= lon <= east and south <= lat <= north

    for obj in osmium.FileProcessor(str(path)).with_locations(index_type):
        tags = obj.tags
        try:
            if obj.is_node():
                for key, values in EMITTER_TAGS.items():
                    value = tags.get(key)
                    if value in values and inside(obj.location.lon, obj.location.lat):
                        out.emitter_points.append((obj.location.lon, obj.location.lat))
                        out.emitter_kinds.append(f"{key}={value}")
                        n_seen += 1
                continue
            if not obj.is_way():
                continue

            coords = [(n.location.lon, n.location.lat) for n in obj.nodes if n.location.valid()]
            if len(coords) < 2:
                continue
            centroid = (float(np.mean([c[0] for c in coords])),
                        float(np.mean([c[1] for c in coords])))
            if not inside(*centroid):
                continue

            for key, values in EMITTER_TAGS.items():
                value = tags.get(key)
                if value in values:
                    out.emitter_points.append(centroid)
                    out.emitter_kinds.append(f"{key}={value}")
                    n_seen += 1

            closed = len(coords) >= 4 and coords[0] == coords[-1]
            landuse = tags.get("landuse")
            if closed and landuse in LANDUSE_VALUES:
                out.landuse_polys.append((landuse, coords))
                n_seen += 1
            if closed and tags.get("building") is not None:
                out.building_points.append(centroid)
                out.building_m2.append(_polygon_area_m2(coords, project))
                n_seen += 1

            highway = tags.get("highway")
            if highway in ROAD_VALUES:
                xy = project(coords)
                seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                mid = (np.asarray(coords[:-1]) + np.asarray(coords[1:])) / 2.0
                for length, point in zip(seg, mid):
                    if inside(point[0], point[1]):
                        out.road_points.append((float(point[0]), float(point[1])))
                        out.road_kinds.append(f"highway={highway}")
                        out.road_m.append(float(length))
                        n_seen += 1
        except (osmium.InvalidLocationError, RuntimeError):
            continue

    logging.info("  %s: %d in-envelope features (%.0f s)", path.name, n_seen,
                 time.time() - started)
    return out


def build_matrix(tiles: list[TileBox], data: CityHarvest, project) -> np.ndarray:
    names = feature_names()
    col = {name: i for i, name in enumerate(names)}
    mat = np.zeros((len(tiles), len(names)), dtype=np.float64)

    if data.emitter_points:
        src, dst = assign_tiles(tiles, np.asarray(data.emitter_points))
        for p, t in zip(src, dst):
            mat[t, col[f"emitter:{data.emitter_kinds[p]}"]] += 1
            mat[t, col["emitter:total"]] += 1

    if data.building_points:
        src, dst = assign_tiles(tiles, np.asarray(data.building_points))
        for p, t in zip(src, dst):
            mat[t, col["building:count"]] += 1
            mat[t, col["building:total_m2"]] += data.building_m2[p]

    if data.landuse_polys:
        # Landuse polygons are intersected against the tile grid rather than charged whole to
        # their centroid's tile. Centroid charging let a single polygon spanning several tiles
        # report a coverage fraction of 8.8 on New York, which ridge treated as a high-leverage
        # feature and overfitted (bottom-90% R2 fell to -1.23). All geometry here is in metres,
        # so intersection areas come straight off `.area`.
        from shapely import STRtree, box
        from shapely.geometry import Polygon

        tile_boxes_m = []
        for t in tiles:
            (x0, y0), (x1, y1) = project([(t.west, t.south), (t.east, t.north)])
            tile_boxes_m.append(box(x0, y0, x1, y1))
        tree = STRtree(tile_boxes_m)
        for value, coords in data.landuse_polys:
            poly = Polygon(project(coords))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            column = col[f"landuse_frac:{value}"]
            for t in tree.query(poly, predicate="intersects"):
                mat[t, column] += poly.intersection(tile_boxes_m[t]).area

    if data.road_points:
        src, dst = assign_tiles(tiles, np.asarray(data.road_points))
        for p, t in zip(src, dst):
            mat[t, col[f"road_m:{data.road_kinds[p].split('=')[1]}"]] += data.road_m[p]
            mat[t, col["road_m:total"]] += data.road_m[p]

    # Landuse columns become tile-area FRACTIONS so they are comparable across tiles and cities.
    tile_area = np.array([
        _polygon_area_m2([(t.west, t.south), (t.east, t.south), (t.east, t.north),
                          (t.west, t.north), (t.west, t.south)], project)
        for t in tiles])
    for value in LANDUSE_VALUES:
        column = col[f"landuse_frac:{value}"]
        # Clip at 1.0: overlapping polygons of the same class can double-count a tile, and a tile
        # cannot be more than fully covered. A remaining excess means the intersection is wrong.
        mat[:, column] = np.clip(mat[:, column] / np.maximum(tile_area, 1.0), 0.0, 1.0)
    counts = mat[:, col["building:count"]]
    mat[:, col["building:mean_m2"]] = np.divide(mat[:, col["building:total_m2"]],
                                                np.maximum(counts, 1.0))
    return mat


def drop_unestimable(matrix: np.ndarray, names: list[str],
                     min_nonzero_frac: float) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove columns active in too few tiles to fit a coefficient on.

    A column nonzero in 2 of 1091 tiles (man_made=gasometer on New York) standardises those two
    tiles to about +-23 and hands ridge a coefficient estimated from a single observation, which
    then transfers arbitrarily across splits. This uses only X, never the targets, and is applied
    once for the whole corpus so the feature definition does not vary by split.
    """
    keep = [(matrix[:, i] > 0).mean() >= min_nonzero_frac for i in range(matrix.shape[1])]
    dropped = [n for n, k in zip(names, keep) if not k]
    return matrix[:, keep], [n for n, k in zip(names, keep) if k], dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", nargs="+", default=list(CITY_EXTRACT))
    parser.add_argument("--index-type", default="flex_mem",
                        help="pyosmium node-location index; sparse_file_array is slower but "
                             "uses far less RAM")
    parser.add_argument("--min-nonzero-frac", type=float, default=0.02,
                        help="drop columns active in fewer than this fraction of tiles; 0.02 "
                             "keeps power=plant (4%% of New York tiles, ~45 tiles, estimable) "
                             "and drops man_made=gasometer (2 tiles, not estimable)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    tiles_by_city = load_tiles(args.cities)
    names = feature_names()
    summary: dict = {"meta": {"features": names,
                              "motivation": "urbanmoe_carbon_anatomy.json -- the carbon "
                                            "error is a clustered, under-predicted top decile; "
                                            "these are ODIAC's own point-source and industrial "
                                            "layers",
                              "limitation": "multipolygon relations are skipped; polygon area is "
                                            "charged to the tile holding its centroid"},
                     "cities": {}}

    for city in args.cities:
        tiles = tiles_by_city[city]
        extract = EXTRACT_DIR / CITY_EXTRACT[city]
        if not extract.exists():
            raise SystemExit(f"{extract} not found. Download the Geofabrik extract first.")
        bounds = (min(t.west for t in tiles), min(t.south for t in tiles),
                  max(t.east for t in tiles), max(t.north for t in tiles))
        logging.info("=== %s: %d tiles, envelope %.3f,%.3f -> %.3f,%.3f ===", city, len(tiles),
                     *bounds)

        project = _projector(tiles)
        data = harvest(extract, bounds, project, args.index_type)
        matrix = build_matrix(tiles, data, project)
        matrix, kept, dropped = drop_unestimable(matrix, names, args.min_nonzero_frac)
        if dropped:
            logging.info("  dropped %d unestimable columns (<%.1f%% of tiles nonzero): %s",
                         len(dropped), 100 * args.min_nonzero_frac, ", ".join(dropped))

        np.save(EMBED_DIR / f"urbanmoe_osm_{city}.npy", matrix)
        np.save(EMBED_DIR / f"urbanmoe_osm_{city}_files.npy",
                np.array([t.name for t in tiles], dtype=object))

        nonzero = (matrix > 0).mean(axis=0)

        def column(name: str) -> np.ndarray:
            return matrix[:, kept.index(name)] if name in kept else np.zeros(len(tiles))

        summary["cities"][city] = {
            "n_tiles": len(tiles),
            "n_features": matrix.shape[1],
            "features_kept": kept,
            "features_dropped": dropped,
            "emitter_tiles_frac": float((column("emitter:total") > 0).mean()),
            "emitter_total": float(column("emitter:total").sum()),
            "max_industrial_frac": float(column("landuse_frac:industrial").max()),
            "p95_industrial_frac": float(np.percentile(column("landuse_frac:industrial"), 95)),
            "median_building_count": float(np.median(column("building:count"))),
            "median_road_km": float(np.median(column("road_m:total")) / 1000),
            "frac_tiles_nonzero_per_feature": {n: round(float(f), 4)
                                               for n, f in zip(kept, nonzero)},
        }
        cell = summary["cities"][city]
        logging.info("  emitters in %.1f%% of tiles (%d total) | industrial frac p95 %.3f max "
                     "%.3f | median %d buildings | median %.1f km road",
                     100 * cell["emitter_tiles_frac"], int(cell["emitter_total"]),
                     cell["p95_industrial_frac"], cell["max_industrial_frac"],
                     int(cell["median_building_count"]), cell["median_road_km"])

    VOCAB_JSON.write_text(json.dumps(
        {"all_features": names,
         "kept_per_city": {c: summary["cities"][c]["features_kept"] for c in summary["cities"]},
         "note": "columns are dropped per city when active in too few tiles; models are trained "
                 "per city so the feature count may differ between cities"},
        indent=2), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote %s and %s", VOCAB_JSON, OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
