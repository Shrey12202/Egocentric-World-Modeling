# Egocentric World Memory Prototype (Flat Layout)

This is a **flat, HPC-friendly codebase** for your project:

> Reconstruct short egocentric clips as NeRF scenes, embed them with CLIP,
> and retrieve "memories" by text query.

Design goals:

- All code lives in **one folder**: `src/`
- Simple imports like `from dataset import Ego4DSequencesIndex`
- No nested packages or tricky PYTHONPATH setup
- Works locally and on NYU HPC by just copying the directory

## Layout

```text
ego_memory_flat_full/
  README.md
  requirements.txt
  src/
    config.py
    dataset_utils.py
    dataset.py
    nerf_model.py
    nerf_training.py
    embedding.py
    memory_bank.py
    prepare_fake_ego4d.py
    train_nerf.py
    build_memories.py
    query_memory.py
```

## Quickstart (local or HPC)

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare a fake Ego4D-style dataset from your own short videos:

```bash
python src/prepare_fake_ego4d.py \
  --raw_video_dir ./raw_videos \
  --output_root   ./data/ego4d_fake
```

3. Train a tiny NeRF on one sequence:

```bash
python src/train_nerf.py \
  --data_root   ./data/ego4d_fake \
  --sequence_id seq_0000 \
  --out_dir     ./checkpoints
```

4. Build a memory bank (NeRF checkpoints + CLIP embeddings):

```bash
python src/build_memories.py \
  --data_root       ./data/ego4d_fake \
  --checkpoints     ./checkpoints \
  --output_memory_db ./memory_db
```

5. Query memories with text:

```bash
python src/query_memory.py \
  --memory_db ./memory_db \
  --query "Where did I see a red mug?"
```

When your **Ego4D** AWS access arrives:

- Download clips such that they look like:

```text
<EGO4D_ROOT>/v1/clips/<sequence_id>/rgb/frame_000000.jpg ...
```

- Then just point `--data_root` to `<EGO4D_ROOT>` and reuse the same scripts.
