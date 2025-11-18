"""A minimal NeRF-like MLP model.

This is **not** a full-featured NeRF implementation, but rather a simple
radiance field MLP that:

- Takes 3D sample points along rays,
- Outputs colors and densities,
- Performs a basic volume rendering integral.

The API is intentionally tiny so you can easily swap in Nerfstudio or
Instant-NGP later while keeping the same training and memory code.
"""  # noqa: D400

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DummyNeRFConfig:
    """Configuration for the dummy NeRF network.

    Attributes
    ----------
    hidden_dim:
        Width of hidden layers.
    num_layers:
        Number of fully-connected layers.
    skips:
        Layers at which to concatenate the original input, mimicking
        NeRF-style skip connections.
    """  # noqa: D401

    hidden_dim: int = 128
    num_layers: int = 4
    skips: Tuple[int, ...] = (2,)


class DummyNeRF(nn.Module):
    """Small MLP approximating a NeRF-like radiance field."""  # noqa: D401

    def __init__(self, cfg: DummyNeRFConfig, in_dim: int = 3):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(cfg.num_layers):
            out_dim = cfg.hidden_dim
            layers.append(nn.Linear(dim, out_dim))
            dim = out_dim
            if i in cfg.skips:
                dim += in_dim
        self.layers = nn.ModuleList(layers)
        self.color_head = nn.Linear(dim, 3)
        self.density_head = nn.Linear(dim, 1)
        self.cfg = cfg

    def forward_mlp(self, x: torch.Tensor):
        """Forward pass through the MLP before rendering.

        Parameters
        ----------
        x:
            Input 3D points, shape (N, 3).

        Returns
        -------
        rgb:
            Predicted colors in [0, 1], shape (N, 3).
        sigma:
            Predicted densities >= 0, shape (N, 1).
        """  # noqa: D401

        h = x
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(h))
            if i in self.cfg.skips:
                h = torch.cat([h, x], dim=-1)
        sigma = F.relu(self.density_head(h))
        rgb = torch.sigmoid(self.color_head(h))
        return rgb, sigma

    def forward(self, sample_pts: torch.Tensor, ray_d: torch.Tensor):
        """Render colors and depths for a batch of rays.

        Parameters
        ----------
        sample_pts:
            Sampled points along rays, shape (N_rays, N_samples, 3).
        ray_d:
            Ray directions, shape (N_rays, 3). Currently unused in this
            simple dummy implementation but kept for API compatibility.

        Returns
        -------
        rgb:
            Per-ray RGB colors, shape (N_rays, 3).
        depth:
            Approximate depth values per ray, shape (N_rays,).
        """  # noqa: D401

        N_rays, N_samples, _ = sample_pts.shape
        pts_flat = sample_pts.reshape(-1, 3)
        rgb_flat, sigma_flat = self.forward_mlp(pts_flat)
        rgb_flat = rgb_flat.reshape(N_rays, N_samples, 3)
        sigma_flat = sigma_flat.reshape(N_rays, N_samples)

        # Simple volume rendering: constant step size along ray
        deltas = torch.ones_like(sigma_flat) * 0.05
        alphas = 1.0 - torch.exp(-sigma_flat * deltas)
        T = torch.cumprod(
            torch.cat([torch.ones_like(alphas[:, :1]), 1.0 - alphas + 1e-10], dim=-1),
            dim=-1,
        )[:, :-1]
        weights = alphas * T  # (N_rays, N_samples)

        rgb = (weights.unsqueeze(-1) * rgb_flat).sum(dim=1)
        depths = (weights * torch.linspace(0.0, 1.0, N_samples, device=weights.device)).sum(dim=1)
        return rgb, depths
