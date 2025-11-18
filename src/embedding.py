"""CLIP-based embedding utilities.

This module wraps OpenAI CLIP to provide:

- Image embedding via :meth:`EmbeddingModel.embed_image`
- Text embedding via :meth:`EmbeddingModel.embed_text`

Embeddings are L2-normalized so cosine similarity == dot product.
"""  # noqa: D400

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from PIL import Image

import clip


@dataclass
class EmbeddingConfig:
    """Configuration for the CLIP embedding model.

    Attributes
    ----------
    model_name:
        Name of the CLIP model to load, e.g. "ViT-B/32".
    device:
        Device string, typically "cuda" or "cpu".
    """  # noqa: D401

    model_name: str = "ViT-B/32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel:
    """Thin wrapper around OpenAI CLIP for image/text embeddings."""  # noqa: D401

    def __init__(self, cfg: EmbeddingConfig):
        self.device = cfg.device
        self.model, self.preprocess = clip.load(cfg.model_name, device=self.device)

    @torch.no_grad()
    def embed_image(self, img: Image.Image) -> torch.Tensor:
        """Compute a CLIP embedding for a single image.

        Parameters
        ----------
        img:
            PIL Image in RGB mode.

        Returns
        -------
        torch.Tensor
            L2-normalized embedding vector on CPU, shape (D,).
        """  # noqa: D401

        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu()

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> torch.Tensor:
        """Compute CLIP embeddings for a list of text prompts.

        Parameters
        ----------
        texts:
            List of strings.

        Returns
        -------
        torch.Tensor
            L2-normalized embeddings on CPU, shape (N, D).
        """  # noqa: D401

        tokens = clip.tokenize(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()
