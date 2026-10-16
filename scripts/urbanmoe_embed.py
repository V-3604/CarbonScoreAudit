"""Embed the benchmark tiles with a frozen SigLIP encoder.

Produces the image features every probe here reads. The encoder is
google/siglip-so400m-patch14-384, used frozen: pooler_output, L2-normalised, 1152-d. Nothing is
fine-tuned anywhere in this analysis.

Expects the released benchmark data under data/raw/urbanmoe/:
    {city}/.../tile*.tif            200x200 RGB uint8 GeoTIFF
    {City}_merged_normalized.csv    filename, c_value, pop_value, light_value

Two quirks of that release are handled here rather than downstream. The CSV filename column
carries a doubled ".tif.tif" while the file on disk has one. The third city is named "New York"
in the released files although the benchmark paper describes Washington DC; the released data is
used as published.

Input : data/raw/urbanmoe/
Output: artifacts/embeddings/urbanmoe_{city}.npy

Run:
    python scripts/urbanmoe_embed.py
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/raw/urbanmoe"
EMBED_DIR = REPO_ROOT / "artifacts/embeddings"

# Mirrors extract_siglip_embeddings.py (same model / pooler_output / L2-norm), but with a
# device-aware dtype so the embed also runs on CPU (fp16 matmul is unsupported/slow on CPU).
MODEL_ID = "google/siglip-so400m-patch14-384"
EMBED_DIM = 1152


def load_model(device: torch.device) -> tuple[AutoModel, AutoProcessor]:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logging.info("Loading %s (%s) …", MODEL_ID, dtype)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).vision_model.to(device).eval()
    return model, processor


@torch.inference_mode()
def embed_batch(images: list[Image.Image], model: AutoModel, processor: AutoProcessor,
                device: torch.device) -> np.ndarray:
    """L2-normalised pooled embeddings, (N, 1152) float32."""
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device, dtype=next(model.parameters()).dtype)
    embs = model(pixel_values=pixel_values).pooler_output.float().cpu().numpy()
    return embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8, None)

# our label -> (extracted folder name, label CSV name). Tile dir is auto-discovered.
CITIES: dict[str, tuple[str, str]] = {
    "shanghai": ("shanghai", "shanghai_merged_normalized.csv"),
    "beijing": ("beijing", "Beijing_merged_normalized.csv"),
    "newyork": ("NewWork", "NewYork_merged_normalized.csv"),
}


def resolve_tile_map(city_folder: Path) -> dict[str, Path]:
    """basename -> path for every .tif under the city folder (tile dir varies per city)."""
    return {p.name: p for p in city_folder.rglob("*.tif")}


def actual_tile_name(csv_filename: str) -> str:
    """CSV stores a doubled extension ('tile_10.tif.tif'); the file is 'tile_10.tif'."""
    return csv_filename[:-4] if csv_filename.endswith(".tif.tif") else csv_filename


def embed_city(city: str, model, processor, device: torch.device, batch_size: int,
               overwrite: bool) -> None:
    import pandas as pd

    folder_name, csv_name = CITIES[city]
    city_folder = DATA_DIR / folder_name
    csv_path = city_folder / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Extract the UrbanMoE zip into {DATA_DIR} first "
            f"(unzip {folder_name}.zip).")

    out_emb = EMBED_DIR / f"urbanmoe_siglip_{city}.npy"
    out_files = EMBED_DIR / f"urbanmoe_siglip_{city}_files.npy"
    if out_emb.exists() and not overwrite:
        logging.info("%s embeddings exist (%s) — skipping. Use --overwrite to redo.", city, out_emb.name)
        return

    labels = pd.read_csv(csv_path)
    tile_map = resolve_tile_map(city_folder)
    logging.info("%s: %d CSV rows, %d .tif tiles on disk", city, len(labels), len(tile_map))

    filenames = labels["filename"].astype(str).tolist()
    paths: list[Path] = []
    kept_files: list[str] = []
    missing: list[str] = []
    for fname in filenames:
        tile = tile_map.get(actual_tile_name(fname))
        if tile is None:
            missing.append(fname)
            continue
        paths.append(tile)
        kept_files.append(fname)
    if missing:
        logging.warning("%s: %d CSV rows have no matching tile (e.g. %s)", city, len(missing), missing[:3])

    n = len(paths)
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    t0 = time.time()
    for start in tqdm(range(0, n, batch_size), desc=f"embed/{city}", unit="batch", dynamic_ncols=True):
        batch_paths = paths[start: start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        embs[start: start + len(batch_paths)] = embed_batch(images, model, processor, device)
    logging.info("%s: embedded %d tiles in %.1fs (%.1f ms/tile)",
                 city, n, time.time() - t0, (time.time() - t0) / max(n, 1) * 1000)

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, embs)
    np.save(out_files, np.array(kept_files, dtype=object))
    logging.info("Saved %s (%d, %d)  +  %s", out_emb.name, n, EMBED_DIM, out_files.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed UrbanMoE satellite tiles with frozen SigLIP.")
    parser.add_argument("--city", choices=[*CITIES, "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("CUDA not available — running on CPU (slow; ~30 min for all 3 cities).")
    model, processor = load_model(device)

    cities = list(CITIES) if args.city == "all" else [args.city]
    for city in cities:
        embed_city(city, model, processor, device, args.batch_size, args.overwrite)
    logging.info("UrbanMoE embedding complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
