"""Train a small NeRF on a single Ego4D-like sequence.

This is a thin CLI wrapper around :func:`nerf_training.train_nerf_on_sequence`.

Example
-------

.. code-block:: bash

    python src/train_nerf.py \
      --data_root   ./data/ego4d_fake \
      --sequence_id seq_0000 \
      --out_dir     ./checkpoints
"""  # noqa: D400

from __future__ import annotations

import argparse
from pathlib import Path

from config import NerfConfig
from dataset import Ego4DSequencesIndex
from nerf_training import train_nerf_on_sequence


def main():
    """CLI entry point to train NeRF on one sequence."""  # noqa: D401

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root of Ego4D-like dataset.")
    ap.add_argument("--sequence_id", type=str, required=True, help="Sequence ID to train on.")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./checkpoints",
        help="Directory to store NeRF checkpoints.",
    )
    args = ap.parse_args()

    index = Ego4DSequencesIndex(Path(args.data_root))
    seq_info = index.get(args.sequence_id)

    cfg = NerfConfig()
    result = train_nerf_on_sequence(seq_info, cfg, work_dir=Path(args.out_dir))
    print(f"Finished training. Checkpoint: {result.checkpoint_path}, final_loss={result.final_loss:.4f}")


if __name__ == "__main__":
    main()
