"""High-level NeRF training helper.

This module defines a convenience function
:func:`train_nerf_on_sequence` which:

1. Takes an :class:`Ego4DSequenceInfo` describing one sequence.
2. Builds a :class:`Ego4DSequenceDataset` over its frames.
3. Samples random rays from the images.
4. Trains :class:`DummyNeRF` for a few epochs.
5. Saves a checkpoint that can later be treated as a "memory".

You can later replace the model in :mod:`nerf_model` with a stronger NeRF
without changing this interface.
"""  # noqa: D400

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import NerfConfig
from dataset import Ego4DSequenceDataset, Ego4DSequenceInfo
from nerf_model import DummyNeRF, DummyNeRFConfig


@dataclass
class TrainResult:
    """Simple struct containing training results.

    Attributes
    ----------
    checkpoint_path:
        Path to the saved NeRF checkpoint (.pt file).
    final_loss:
        Final MSE loss value on the last batch.
    """  # noqa: D401

    checkpoint_path: Path
    final_loss: float


def _images_to_rays(rgb_batch: torch.Tensor):
    """Convert a batch of images into ray origins/directions and colors.

    This is a very simplified ray generator that ignores real camera
    intrinsics and just fires rays from the origin through a regular grid.

    Parameters
    ----------
    rgb_batch:
        Tensor with shape (B, 3, H, W) in [0, 1].

    Returns
    -------
    origins:
        Ray origins, shape (B * H * W, 3).
    dirs:
        Ray directions, shape (B * H * W, 3).
    colors:
        Target RGB colors, shape (B * H * W, 3).
    """  # noqa: D401

    B, C, H, W = rgb_batch.shape
    device = rgb_batch.device
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    dirs = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # (H, W, 3)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.view(1, H, W, 3).expand(B, -1, -1, -1)
    dirs = dirs.reshape(B * H * W, 3)
    origins = torch.zeros_like(dirs)
    colors = rgb_batch.permute(0, 2, 3, 1).reshape(B * H * W, 3)
    return origins, dirs, colors


def train_nerf_on_sequence(
    sequence_info: Ego4DSequenceInfo,
    cfg: NerfConfig,
    device: str | None = None,
    work_dir: Path | str = "./checkpoints",
) -> TrainResult:
    """Train a :class:`DummyNeRF` model on a single sequence.

    Parameters
    ----------
    sequence_info:
        Metadata describing which frames to load.
    cfg:
        NeRF training configuration.
    device:
        Device string such as "cuda" or "cpu". If ``None``, picks
        CUDA if available.
    work_dir:
        Directory where the checkpoint will be saved.

    Returns
    -------
    TrainResult
        A small struct containing the checkpoint path and final loss.
    """  # noqa: D401

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    ds = Ego4DSequenceDataset(
        sequence_info,
        img_size=cfg_img_size(),
        frame_stride=cfg_frame_stride(),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    model = DummyNeRF(DummyNeRFConfig()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    last_loss = 0.0
    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        running_loss = 0.0
        for batch in pbar:
            rgb = batch["rgb"].to(device)  # (1, 3, H, W)
            origins, dirs, target = _images_to_rays(rgb)
            origins, dirs, target = origins.to(device), dirs.to(device), target.to(device)

            N_rays = min(cfg.batch_size, origins.shape[0])
            idx = torch.randint(0, origins.shape[0], (N_rays,), device=device)
            origins_b = origins[idx]
            dirs_b = dirs[idx]
            target_b = target[idx]

            t_vals = torch.linspace(cfg.near, cfg.far, cfg.num_samples, device=device)
            sample_pts = origins_b.unsqueeze(1) + dirs_b.unsqueeze(1) * t_vals.view(1, -1, 1)

            pred_rgb, _ = model(sample_pts, dirs_b)
            loss = torch.nn.functional.mse_loss(pred_rgb, target_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            last_loss = loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n or 1):.4f}"})

    ckpt_path = work_dir / f"nerf_{sequence_info.sequence_id}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "sequence_id": sequence_info.sequence_id,
            "cfg": cfg.__dict__,
        },
        ckpt_path,
    )
    return TrainResult(checkpoint_path=ckpt_path, final_loss=float(last_loss))


def cfg_img_size() -> int:
    """Central location to adjust NeRF input image size."""  # noqa: D401

    return 128


def cfg_frame_stride() -> int:
    """Central location to adjust frame subsampling factor."""  # noqa: D401

    return 4
