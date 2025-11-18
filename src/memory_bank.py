"""Scene memory bank using FAISS for retrieval.

Each "memory" corresponds to a reconstructed scene (one sequence) and
consists of:

- a **memory_id** (e.g. sequence_id)
- a CLIP embedding vector (scene descriptor)
- a path to a NeRF checkpoint on disk

The :class:`MemoryBank` stores all embeddings in a FAISS index and
metadata in a Python list. You can save/load the whole bank to/from disk.
"""  # noqa: D400

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import json


@dataclass
class MemoryEntry:
    """One memory entry representing a single scene.

    Attributes
    ----------
    memory_id:
        Identifier of the memory, typically a sequence_id.
    embedding:
        Numpy array with shape (D,) representing the scene.
    checkpoint_path:
        Path to the NeRF checkpoint file for this scene.
    """  # noqa: D401

    memory_id: str
    embedding: np.ndarray
    checkpoint_path: Path


class MemoryBank:
    """In-memory database of scene memories with FAISS-based search."""  # noqa: D401

    def __init__(self, dim: int, index_factory: str = "Flat"):
        self.dim = dim
        self.index = faiss.index_factory(dim, index_factory)
        self.entries: List[MemoryEntry] = []

    def add(self, memory_id: str, embedding: np.ndarray, checkpoint_path: Path):
        """Add a new memory to the bank and FAISS index.

        Parameters
        ----------
        memory_id:
            Unique identifier for this scene.
        embedding:
            Numpy array with shape (D,).
        checkpoint_path:
            Path to the NeRF checkpoint file.
        """  # noqa: D401

        if embedding.shape[-1] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {embedding.shape[-1]}")
        self.entries.append(
            MemoryEntry(memory_id, embedding.astype("float32"), checkpoint_path)
        )
        self.index.add(embedding.astype("float32")[None, :])

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """Search for the top-k closest memories to a query embedding.

        Parameters
        ----------
        query:
            Query vector, shape (D,).
        k:
            Number of nearest neighbors to return.

        Returns
        -------
        List[Tuple[MemoryEntry, float]]
            List of (entry, distance) tuples sorted by increasing distance.
        """  # noqa: D401

        if query.shape[-1] != self.dim:
            raise ValueError(f"Expected query dim {self.dim}, got {query.shape[-1]}")
        D, I = self.index.search(query.astype("float32")[None, :], k)
        results: List[Tuple[MemoryEntry, float]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.entries):
                continue
            results.append((self.entries[idx], float(dist)))
        return results

    def save(self, path: Path | str):
        """Save FAISS index and metadata to ``path`` directory."""  # noqa: D401

        path = Path(path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata as JSON
        meta = []
        for e in self.entries:
            meta.append(
                {
                    "memory_id": e.memory_id,
                    "embedding": e.embedding.tolist(),
                    "checkpoint_path": str(e.checkpoint_path),
                }
            )
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path | str) -> "MemoryBank":
        """Load a :class:`MemoryBank` previously saved with :meth:`save`."""  # noqa: D401

        path = Path(path).expanduser().resolve()
        meta = json.loads((path / "meta.json").read_text())
        if not meta:
            raise RuntimeError("Empty memory bank meta.json")
        dim = len(meta[0]["embedding"])
        bank = cls(dim=dim)
        bank.index = faiss.read_index(str(path / "index.faiss"))
        bank.entries = [
            MemoryEntry(
                memory_id=m["memory_id"],
                embedding=np.array(m["embedding"], dtype="float32"),
                checkpoint_path=Path(m["checkpoint_path"]),
            )
            for m in meta
        ]
        return bank
