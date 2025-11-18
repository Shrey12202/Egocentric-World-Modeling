"""Global configuration dataclasses.

These small config objects keep hyperparameters and path-related defaults
in one place so you can pass them around cleanly and, if needed, log them
for reproducibility.
"""  # noqa: D400

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing.

    Attributes
    ----------
    root:
        Root directory of an Ego4D-like dataset. Expected structure:

        root/
          v1/
            clips/
              <sequence_id>/
                rgb/frame_000000.jpg
    use_clips:
        Whether to use the `v1/clips/` directory (True) or something else.
        For now we assume clips.
    sequence_len:
        Unused in the basic prototype but can be used if you want to sample
        fixed-length temporal chunks.
    frame_stride:
        Subsampling factor when reading frames (every N-th frame).
    img_size:
        Target size (square) for images fed into NeRF.
    """  # noqa: D401

    root: Path
    use_clips: bool = True
    sequence_len: int = 32
    frame_stride: int = 2
    img_size: int = 128


@dataclass
class NerfConfig:
    """Configuration for NeRF training.

    Attributes
    ----------
    num_epochs:
        Number of passes over the sequence frames.
    lr:
        Learning rate for Adam optimizer.
    batch_size:
        Number of rays to sample per optimization step.
    near, far:
        Near/far plane distances for sampling along rays.
    num_samples:
        Number of samples per ray.
    """  # noqa: D401

    num_epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 1024
    near: float = 0.5
    far: float = 5.0
    num_samples: int = 64


@dataclass
class MemoryConfig:
    """Configuration for the memory bank and embeddings.

    Attributes
    ----------
    embedding_dim:
        Dimensionality of the CLIP embeddings (e.g. 512 for ViT-B/32).
    index_factory:
        FAISS index factory string, e.g. "Flat" or "IVF100,Flat".
    device:
        Device string for running CLIP ("cuda" or "cpu").
    """  # noqa: D401

    embedding_dim: int = 512
    index_factory: str = "Flat"
    device: str = "cuda"
