"""Create a fake Ego4D-like dataset from local videos.

This script is useful *before* you have official Ego4D access. It turns a
folder of short videos into the directory layout expected by
:class:`dataset.Ego4DSequencesIndex`.

Example
-------

.. code-block:: bash

    python src/prepare_fake_ego4d.py \
      --raw_video_dir ./raw_videos \
      --output_root   ./data/ego4d_fake

Each video in ``raw_video_dir`` becomes one sequence:

.. code-block:: text

    output_root/v1/clips/seq_0000/rgb/frame_000000.jpg
"""  # noqa: D400

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video_path: Path, out_dir: Path, max_frames: int = 300, stride: int = 2):
    """Extract frames from a video into ``out_dir``.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    out_dir:
        Directory where frames will be saved as JPGs.
    max_frames:
        Maximum number of frames to extract.
    stride:
        Keep every ``stride``-th frame to subsample the video.
    """  # noqa: D401

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            # OpenCV returns BGR; convert to RGB for consistency then back to BGR for saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_path = out_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()


def main():
    """CLI entry point for preparing a fake Ego4D-style dataset."""  # noqa: D401

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_video_dir", type=str, required=True, help="Directory with input videos.")
    ap.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for Ego4D-like structure (will create v1/clips).",
    )
    ap.add_argument("--max_frames", type=int, default=300, help="Max frames per video.")
    ap.add_argument("--stride", type=int, default=2, help="Frame stride when sampling.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_video_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    clips_root = output_root / "v1" / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    # Collect videos with common extensions
    videos = sorted(
        [
            p
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv")
            for p in raw_dir.glob(ext)
        ]
    )
    if not videos:
        raise RuntimeError(f"No videos found in {raw_dir}")

    for i, vpath in enumerate(tqdm(videos, desc="Preparing fake Ego4D")):
        seq_id = f"seq_{i:04d}"
        out_rgb_dir = clips_root / seq_id / "rgb"
        extract_frames(vpath, out_rgb_dir, max_frames=args.max_frames, stride=args.stride)

    print(f"Done. Fake Ego4D-like dataset written to {output_root}")


if __name__ == "__main__":
    main()
