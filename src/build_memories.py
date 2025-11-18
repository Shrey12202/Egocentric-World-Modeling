"""Build a memory bank from Ego4D-like sequences.

For each sequence, this script will:

1. Train (or reuse) a NeRF checkpoint.
2. Take one representative frame (first frame).
3. Embed the image using CLIP.
4. Add an entry to :class:`memory_bank.MemoryBank`.

Result: a directory containing a FAISS index and a JSON metadata file.

Example
-------

.. code-block:: bash

    python src/build_memories.py \
      --data_root        ./data/ego4d_fake \
      --checkpoints      ./checkpoints \
      --output_memory_db ./memory_db
"""  # noqa: D400

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from config import NerfConfig, MemoryConfig
from dataset import Ego4DSequencesIndex, Ego4DSequenceDataset
from nerf_training import train_nerf_on_sequence
from embedding import EmbeddingModel, EmbeddingConfig
from memory_bank import MemoryBank


def main():
    """CLI entry point to build a scene memory bank."""  # noqa: D401

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root of Ego4D-like dataset.")
    ap.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints",
        help="Directory to store or reuse NeRF checkpoints.",
    )
    ap.add_argument(
        "--output_memory_db",
        type=str,
        required=True,
        help="Directory where the memory bank (FAISS + JSON) will be saved.",
    )
    ap.add_argument(
        "--reuse_checkpoints",
        action="store_true",
        help="If set, do not retrain NeRF if a checkpoint already exists.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    ckpt_root = Path(args.checkpoints).expanduser().resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    index = Ego4DSequencesIndex(data_root)
    emb_model = EmbeddingModel(EmbeddingConfig())
    mem_cfg = MemoryConfig()
    bank = MemoryBank(dim=mem_cfg.embedding_dim, index_factory=mem_cfg.index_factory)

    # Loop over sequences, training NeRFs and creating memories
    for seq_id in index.list_sequences():
        seq_info = index.get(seq_id)
        ckpt_path = ckpt_root / f"nerf_{seq_id}.pt"

        if not (args.reuse_checkpoints and ckpt_path.exists()):
            print(f"[NeRF] Training sequence {seq_id}...")
            train_nerf_on_sequence(seq_info, NerfConfig(), work_dir=ckpt_root)
        else:
            print(f"[NeRF] Reusing existing checkpoint for {seq_id}")

        ds = Ego4DSequenceDataset(seq_info, img_size=256, frame_stride=4)
        # Use the first frame as a simple representative of the scene
        item0 = ds[0]
        img_tensor = item0["rgb"]  # (3, H, W)
        img = Image.fromarray(
            (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        )

        emb = emb_model.embed_image(img).numpy()  # (D,)
        if emb.shape[0] != mem_cfg.embedding_dim:
            # If the dimensionality differs (e.g. you changed CLIP model),
            # recreate the memory bank with the new dimension.
            mem_cfg.embedding_dim = emb.shape[0]
            bank = MemoryBank(dim=mem_cfg.embedding_dim, index_factory=mem_cfg.index_factory)
        bank.add(seq_id, emb, ckpt_path)

    bank.save(Path(args.output_memory_db))
    print(f"Memory bank saved to {args.output_memory_db}")


if __name__ == "__main__":
    main()
