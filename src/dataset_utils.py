"""Utility transforms for image preprocessing.

This file intentionally reimplements a minimal subset of transforms instead
of depending on `torchvision.transforms`. That makes it easier to run in
lean environments (like HPC) and keeps behavior explicit.

All transforms operate on PIL Images and most are composed via :class:`Compose`.
"""  # noqa: D400

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from PIL import Image
import torch


class Compose:
    """Compose several transforms together.

    Parameters
    ----------
    transforms:
        List of callables that take and return a PIL Image or tensor.
    """  # noqa: D401

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


@dataclass
class ResizeShortestEdge:
    """Resize image so that the shortest edge equals ``size``.

    The aspect ratio is preserved; only the overall scale changes.
    """  # noqa: D401

    size: int

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if min(w, h) == self.size:
            return img
        if w < h:
            new_w = self.size
            new_h = int(h * self.size / w)
        else:
            new_h = self.size
            new_w = int(w * self.size / h)
        return img.resize((new_w, new_h), resample=Image.BILINEAR)


@dataclass
class CenterCropSquare:
    """Crop image to a central square and then resize to ``size x size``."""  # noqa: D401

    size: int

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        if side != self.size:
            img = img.resize((self.size, self.size), resample=Image.BILINEAR)
        return img


class ToTensor01:
    """Convert a PIL Image to a float32 torch tensor in [0, 1], shape (C, H, W)."""  # noqa: D401

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype("float32") / 255.0
        # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)
