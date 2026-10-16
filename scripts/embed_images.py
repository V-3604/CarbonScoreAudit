"""Embed the benchmark tiles with a frozen SigLIP image encoder.

The encoder is google/siglip-so400m-patch14-384, used frozen: the pooled output of the vision
tower, L2-normalised, 1152 dimensions. Rows follow the label CSV, and only tiles present on disk
are kept.

Output: artifacts/embeddings/siglip_{city}.npy
        artifacts/embeddings/siglip_{city}_files.npy

Run:
    python scripts/embed_images.py
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from common import CITIES, EMBED_DIR, LABEL_CSV, RAW_DIR, TILE_DIR, configure_logging

MODEL_ID = "google/siglip-so400m-patch14-384"
EMBED_DIM = 1152


def load_model(device: torch.device) -> tuple[torch.nn.Module, AutoProcessor]:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).vision_model.to(device).eval()
    return model, processor


@torch.inference_mode()
def embed_batch(images: list[Image.Image], model: torch.nn.Module, processor: AutoProcessor,
                device: torch.device) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device, dtype=next(model.parameters()).dtype)
    embeddings = model(pixel_values=pixel_values).pooler_output.float().cpu().numpy()
    return embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)


def embed_city(city: str, model: torch.nn.Module, processor: AutoProcessor,
               device: torch.device, batch_size: int) -> None:
    csv_path = RAW_DIR / LABEL_CSV[city]
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Extract the benchmark release into {RAW_DIR}.")
    on_disk = {path.name: path for path in (RAW_DIR / TILE_DIR[city]).glob("*.tif")}

    files, paths = [], []
    for name in pd.read_csv(csv_path)["filename"].astype(str):
        # The CSV names tiles 'tile_N.tif.tif'; the file on disk is 'tile_N.tif'.
        path = on_disk.get(name[:-4] if name.endswith(".tif.tif") else name)
        if path is None:
            logging.warning("%s: no tile on disk for %s", city, name)
            continue
        files.append(name)
        paths.append(path)

    embeddings = np.zeros((len(paths), EMBED_DIM), dtype=np.float32)
    start_time = time.time()
    for start in tqdm(range(0, len(paths), batch_size), desc=city, unit="batch"):
        batch = [Image.open(p).convert("RGB") for p in paths[start:start + batch_size]]
        embeddings[start:start + len(batch)] = embed_batch(batch, model, processor, device)
    logging.info("%-9s %d tiles in %.0f s", city, len(paths), time.time() - start_time)

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBED_DIR / f"siglip_{city}.npy", embeddings)
    np.save(EMBED_DIR / f"siglip_{city}_files.npy", np.array(files, dtype=object))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    configure_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("No GPU found; embedding all three cities on CPU takes about 30 min.")
    model, processor = load_model(device)
    for city in CITIES:
        embed_city(city, model, processor, device, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
