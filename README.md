
# O'Prior: Synthetic Prior Generation for Tabular Foundation Models

This repository implements **O'Prior**, a configurable generator of synthetic tabular tasks (structural causal models, realism transforms, and distribution-shift stressors) used to study and pre-train tabular foundation models via in-context learning (ICL).

## Overview

Tabular foundation models are trained on large collections of synthetic datasets sampled from a **prior** over tasks. O'Prior parameterizes that prior along three axes emphasized in the paper:

1. **SCM family mix** — linear, MLP, convolutional, tree-based, Gaussian-process, time-lagged, and hybrid structural causal models, combined with optional hierarchical (graph-aware) sampling.
2. **Realism** — staged corruption and structure (missingness, heavy tails, categoricals, rank normalization, fingerprints, etc.) controlled by `low` / `mild` / `hard` profiles and optional realism curriculum.
3. **Distribution shift** — query-only covariate shift, seasonal drift, and temporal drift for robustness experiments.

The main entry point is batch generation via `o_prior.prior.genload`, which writes serialized episodes to disk (PyTorch `.pt` or HDF5 `.h5`).

## Installation

Requires **Python 3.9–3.12** and a PyTorch build compatible with your CUDA setup (CPU-only is supported for small smoke tests).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

For HDF5 output (`--save_format h5`), install `h5py` if it is not already pulled in:

```bash
pip install h5py
```

Optional: GPU-accelerated tree fitting via [RAPIDS cuML](https://docs.rapids.ai/) when generating large tree-SCM batches on CUDA.

## Quick start

Generate a single batch to `./data` (defaults: mixed SCM prior, PyTorch format):

```bash
python -m o_prior.prior.genload \
  --save_dir ./data \
  --num_batches 1 \
  --batch_size 32 \
  --max_features 20 \
  --max_seq_len 256 \
  --device cpu
```

### Prior types (`--prior_type`)

| Value | Description |
|-------|-------------|
| `mlp_scm`, `conv_scm`, `tree_scm`, `gp_scm`, `linear_scm`, `time_lagged_scm`, `hybrid_scm` | Single SCM family |
| `mix_scm` | Weighted mixture of SCM families |
| `mix_scm_hscm` | Mixture with hybrid / graph-aware hybrid SCM (default in scripts) |
| `mix_scm_no_gp`, `mix_scm_hscm_no_gp` | Mix variants without GP component |

### Realism (`--realism_profile`, `--use_realism_curriculum`)

- **Profiles:** `low`, `mild`, `hard` — control rates of missingness, heavy tails, categorical injection, cross-sectional rank, etc. (see `prior_config.py`).
- **Curriculum:** `--use_realism_curriculum True` with `--realism_profile_start` / `--realism_profile_end` interpolates difficulty over batches.

### Feature sampling (`--sampling`)

`normal`, `uniform`, `mixed`, or `beta` (recommended in paper scripts for feature-count diversity).

### Output formats

- `pt` — PyTorch serialized batches (default).
- `h5` — HDF5 with compressed tensors (used in reproduction scripts).

## Reproducing paper prior configurations

Three shell scripts mirror the main experimental prior settings. Set `DATA_ROOT` to your output directory before running.

### RQ1(d) — core SCM mix, low realism

```bash
export DATA_ROOT=/path/to/output/rq1d
bash scripts/generate_rq1d.sh
```

Defaults: `prior_type=mix_scm_hscm`, `realism_profile=low`, realism transforms mostly off, `max_features=50`, classification (`max_classes=10`).

### RQ2(c) — hard realism

```bash
export DATA_ROOT=/path/to/output/rq2c
bash scripts/generate_rq2c.sh
```

Defaults: `realism_profile=hard`, skewness / SVD / fingerprint / categorical / cross-sectional rank enabled, `hybrid_sampling_strategy=graph_aware`.

### RQ4 — full O'Prior (curricula + shifts)

```bash
export DATA_ROOT=/path/to/output/rq4
bash scripts/generate_rq4.sh
```

Defaults: feature and realism **curricula**, covariate / seasonal / temporal drift enabled, `max_features=100`, `hybrid_sampling_strategy=graph_aware`.

All scripts support environment overrides (e.g. `DEVICE=cuda`, `NUM_BATCHES`, `RESUME_FROM`, `SAVE_FORMAT=h5`). Logs are written under `$DATA_ROOT/logs/`.

## Direct CLI usage (custom priors)

```bash
python -m o_prior.prior.genload \
  --save_dir /path/to/batches \
  --save_format h5 \
  --prior_type mix_scm_hscm \
  --realism_profile hard \
  --use_realism_curriculum True \
  --realism_profile_start mild \
  --realism_profile_end hard \
  --apply_covariate_shift True \
  --apply_temporal_drift True \
  --sampling beta \
  --num_batches 500 \
  --batch_size 50 \
  --device cuda
```

Run `python -m o_prior.prior.genload --help` for the full flag list (tree prior weights, finance stages, missingness, warping, etc.).

## Loading batches for training

`LoadPriorDataset` and `MultiTaskEpisodeIterableDataset` in `episodes.py` read generated batch folders and stream ICL episodes (support/query splits) for downstream training code (not included in this release).

```python
from o_prior.prior.genload import LoadPriorDataset

loader = LoadPriorDataset(
    data_dir="/path/to/batches",
    batch_size=512,
    loop=True,
)
for batch in loader:
    # batch contains X, y, metadata …
    break
```

## Scope of this release

This repository contains **synthetic prior generation** only. Training tabular foundation models, evaluation on OpenML/CC18, and plotting are described in the paper but not shipped here.

## License

MIT License — see [LICENSE](LICENSE). Copyright holder is anonymized during review.
