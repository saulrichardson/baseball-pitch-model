# Setup

This project is designed to be run on:
- **Greene** (recommended for scale; GPU training, large parquet, `$VAST` artifacts)
- a local machine (OK for smoke tests and small slices)

## Install (pip + editable)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Configure artifact root

All commands write outputs (raw downloads, prepared parquet, runs, exported bundles) under an artifact root:

```bash
export BASEBALL_ARTIFACT_ROOT="/path/to/artifacts"
```

On Greene, the intended default is:

```bash
export BASEBALL_ARTIFACT_ROOT="$VAST/baseball_pitch_model"
```

## First end-to-end run (small)

### 1) Download Statcast data

Example: April–May 2023.

```bash
python -m baseball download --start 2023-04-01 --end 2023-05-31
```

### 2) Prepare a modeling dataset

```bash
python -m baseball prepare \
  --start 2023-04-01 \
  --end 2023-05-31 \
  --history-len 8 \
  --min-pitcher-count 1 \
  --min-batter-count 1
```

### 3) Train

Baseline sanity model:

```bash
python -m baseball train --model baseline_mlp
```

Transformer + MDN (pitch type + location):

```bash
python -m baseball train --model transformer_mdn
```

State-capable transformer (required for open-loop rollouts):

```bash
python -m baseball train --model transformer_mdn_state_mt
```

### 4) Evaluate

```bash
python -m baseball eval --split valid
```

### 5) Generate a report (recommended)

```bash
python -m baseball report --run-id <RUN_ID> --split valid
```

### 6) Simulate held-out games (optional)

```bash
python -m baseball simulate --run-id <RUN_ID> --split valid --mode replay --max-games 5
python -m baseball simulate --run-id <RUN_ID> --split valid --mode rollout --count-mode clamp --max-games 5 --device cuda
```

### 7) Export + Serve (optional)

```bash
python -m baseball export --run-id <RUN_ID>
python -m baseball serve --host 0.0.0.0 --port 8000
```

## Tests

```bash
pytest -q
```

