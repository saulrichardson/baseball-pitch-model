# Baseball: Pitch Type + Location Model

This repo builds an end-to-end system to **predict (1) pitch type** and **(2) pitch location** from pre-pitch context + pitch sequence history.

Key constraints from this project:
- Train + evaluate at scale on the **Greene** cluster.
- Store all large artifacts (data, runs, checkpoints) under **`$VAST`**.
- Prefer explicit, reproducible pipelines over ad-hoc notebooks.

## Quick start (Greene)

### 1) Create environment

On Greene:

```bash
cd /vast/$USER
git clone <this repo> baseball
cd baseball

module load anaconda3/2024.02
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 2) Configure artifact root

All commands require an artifact root. On Greene, you already have `$VAST`:

```bash
export BASEBALL_ARTIFACT_ROOT="$VAST/baseball_pitch_model"
```

### 3) Download Statcast pitch-level data

Example: download April–May 2023 (small-ish) for iteration.

```bash
python -m baseball download \
  --start 2023-04-01 \
  --end 2023-05-31
```

### 4) Build a modeling dataset

```bash
python -m baseball prepare \
  --start 2023-04-01 \
  --end 2023-05-31 \
  --history-len 8 \
  --min-pitcher-count 1 \
  --min-batter-count 1
```

### 5) Train models

Baseline (context-only):
```bash
python -m baseball train --model baseline_mlp
```

Sequence model (Transformer + MDN for location):
```bash
python -m baseball train --model transformer_mdn
```

Multi-task sequence model (adds pitch outcome/description head; useful for rollouts and often improves representations):
```bash
python -m baseball train --model transformer_mdn_mt
```

State-capable multi-task model (predicts next game state + pitch description; enables open-loop simulation):
```bash
python -m baseball train --model transformer_mdn_state_mt
```

### 6) Evaluate

```bash
python -m baseball eval --split valid
```

### 6b) Generate a full report (recommended)

The `report` command runs a richer evaluation that is designed for writeups:
- overall metrics (pitch type + location)
- **calibration** (ECE + bins) for pitch type probabilities
- **slice metrics** (count / inning / runners / score / batter stance / pitcher handedness)
- **empirical baselines** fitted on train (pitcher-only, pitcher+count, pitcher+count+prev pitch, etc)

```bash
python -m baseball report --run-id <RUN_ID> --split valid
```

By default this writes JSON artifacts under:

```
$BASEBALL_ARTIFACT_ROOT/runs/<RUN_ID>/report/
```

### 7) Export a deployment bundle

Creates a self-contained directory under `$BASEBALL_ARTIFACT_ROOT/models/exported/<run_id>/`
containing `best.pt`, `meta.json`, and vocab files.

```bash
python -m baseball export
```

### 8) Serve predictions (FastAPI)

```bash
python -m baseball serve --host 0.0.0.0 --port 8000
```

Then `POST /predict` with a `PredictRequest` JSON body.

## Project layout (under `$BASEBALL_ARTIFACT_ROOT`)

```
data/
  raw/            # cached downloads
  prepared/       # parquet datasets + vocabularies
runs/
  <run_id>/
    config.json
    metrics.jsonl
    checkpoints/
models/
  exported/       # exported inference bundles
```

## Notes

- The data source is MLB Statcast via `pybaseball`.
- The location head is probabilistic (2D Gaussian mixture) so we can score with **negative log-likelihood**, not just MSE.

## Development / tests

This repo includes a small `pytest` suite for key encoding/decoding logic and rollout count constraints.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Outcome-aware AB history (schema v4)

The prepare pipeline can optionally include **previous pitch outcomes** (Statcast `description`) in the within-PA
history. When present, the prepared dataset will contain `hist_description_id_*` columns and a `vocabs/description.json`
vocabulary.

- Models use this automatically when `ModelConfig.n_descriptions > 0` (i.e., when the prepared vocab includes `description`).
- If you train a multi-task model (`transformer_mdn_mt` / `transformer_mdn_state_mt`), the model also predicts the
  current pitch `description_id`. In `simulate --mode rollout`, this is used to evolve `hist_desc` instead of leaving it unknown.
- Rollout stability tools:
  - `python -m baseball simulate --mode rollout --count-mode heads` uses the model's state heads directly.
  - `--count-mode clamp` enforces simple within-PA constraints on balls/strikes (no decreases, +1 max, only one increments).
  - `--count-mode rules` evolves balls/strikes from the predicted `description` token (deterministic Statcast rules).
  - `--count-mode constrained` scores rule-consistent transitions vs head transitions (outcome-aware models only).
- Serving (`python -m baseball serve`) accepts `history[].description` for each historical pitch; it is required when the
  loaded model bundle has a description vocab.
- Profiling supports conditioning on previous outcomes:
  - `python -m baseball profile --by pitcher_count_prev_outcome ...`
  - For batter-conditioned profiles, use the less-sparse situation keys first:
    - `pitcher_situation_batter_cluster`
    - `pitcher_situation_batter_cluster_prev`
