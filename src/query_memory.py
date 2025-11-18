"""Query the memory bank using a text prompt.

This script:

1. Loads a previously built memory bank (FAISS + JSON).
2. Embeds the text query with CLIP.
3. Finds the nearest scene memories.
4. Prints the top-k matches.

Example
-------

.. code-block:: bash

    python src/query_memory.py \
      --memory_db ./memory_db \
      --query "Where did I see a red mug?" \
      --top_k 5
"""  # noqa: D400

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embedding import EmbeddingModel, EmbeddingConfig
from memory_bank import MemoryBank


def main():
    """CLI entry point to query the memory bank by text."""  # noqa: D401

    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_db", type=str, required=True, help="Path to memory bank directory.")
    ap.add_argument("--query", type=str, required=True, help="Text query to search for.")
    ap.add_argument("--top_k", type=int, default=5, help="Number of nearest memories to show.")
    args = ap.parse_args()

    bank = MemoryBank.load(Path(args.memory_db))
    emb_model = EmbeddingModel(EmbeddingConfig())

    text_feat = emb_model.embed_text([args.query]).numpy()[0]
    results = bank.search(text_feat, k=args.top_k)

    print(f"Top-{args.top_k} results for query: '{args.query}'\n")
    for entry, dist in results:
        print(f"- memory_id={entry.memory_id}  dist={dist:.4f}  ckpt={entry.checkpoint_path}")


if __name__ == "__main__":
    main()
