# Pitch Type + Location Modeling (Statcast) — Transformer + Simulation

This repository is an end-to-end, artifact-driven pipeline to predict:

1) **Pitch type** (classification)
2) **Pitch location** (probabilistic regression; `plate_x`, `plate_z`)

conditioned on:
- the **pre-pitch game state** (count, outs, inning, runners, score diff, etc.)
- **who** is pitching / hitting (pitcher + batter IDs, handedness / stance)
- the **within-PA pitch history** (sequence of prior pitch types + locations, optionally outcomes)

The design goal is not “a model that scores well in teacher forcing only” — it’s a model that can be:
- evaluated rigorously,
- rolled forward in **open-loop simulation** (to study drift),
- summarized into **pitcher policy profiles** (“what does this pitcher likely do next?”),
- trained efficiently on older GPUs using streaming parquet.

## Data + dataset construction

**Source**: MLB Statcast pitch-by-pitch data fetched via `pybaseball.statcast` (`src/baseball/data/download.py:1`).

**Key engineering point**: Statcast downloads can legitimately return **empty data** (offseason / no games), and those chunks can yield empty-schema parquets. The pipeline treats those as first-class, producing schema-stable empty chunks so that dataset construction doesn’t fail mid-range (`src/baseball/data/download.py:1`, `src/baseball/data/prepare.py:1`, `src/baseball/data/raw_schema.py:1`).

**Prepared dataset**:
- stored as sharded parquet under `$BASEBALL_ARTIFACT_ROOT/data/prepared/{train,valid}/`
- fixed-width history columns (e.g., `hist_pitch_type_id_0..L-1`, `hist_plate_x_0..L-1`, `hist_plate_z_0..L-1`)
- optional outcome-aware history (`hist_description_id_*`) when schema v4 is used

## Models

The repo includes multiple models; the main focus is the state-capable transformer:

- `baseline_mlp`: context + flattened history (sanity baseline)
- `transformer_mdn`: transformer over pitch-history tokens + MDN head for (x,z)
- `transformer_mdn_mt`: adds an outcome/description head (multi-task)
- `transformer_mdn_state_mt`: adds **next-state heads** (balls/strikes/outs/bases/inning transitions) to enable rollouts

Implementation: `src/baseball/training/models.py:1`

### Location as a probabilistic head (MDN)

Pitch location is modeled as a **2D Gaussian mixture** (MDN), so evaluation includes:
- `loc_rmse_ft` (intuitive error in feet)
- `loc_nll` (negative log-likelihood; proper scoring rule for probabilistic predictions)

## Scale + efficiency

This repo is meant to run at scale on the Greene cluster with large artifacts stored under `$VAST`.

Key scaling mechanics:
- training uses a streaming iterable dataset to keep CPU RAM low (`src/baseball/training/streaming.py:1`)
- transformer models support **gradient checkpointing** to reduce activation memory (`src/baseball/training/models.py:1`)
- Slurm scripts live under `scripts/greene/` and default to older GPUs (RTX8000 / V100)

## Evaluation + reporting

Two layers of evaluation:

1) `python -m baseball eval`: lightweight overall metrics (good for fast iteration)
2) `python -m baseball report`: writeup-ready report:
   - overall metrics
   - calibration (ECE bins)
   - slice metrics (count/inning/runners/score/handedness)
   - empirical baselines fitted on train (pitcher-only, pitcher+count, pitcher+count+prev pitch, etc.)

Code: `src/baseball/report.py:1` (CLI: `src/baseball/cli/report.py:1`)

## Open-loop simulation (rollouts)

A key differentiator is explicit support for **open-loop rollouts** over held-out games:

- `simulate --mode replay`: teacher-forced (uses the real pre-pitch state each step)
- `simulate --mode rollout`: feeds predicted next-state back into the next prediction

Rollouts are hard because state errors compound. The code measures this explicitly via `state_curr_match` and provides multiple count-update policies:
- `heads`: trust the model heads
- `clamp`: enforce within-PA constraints on balls/strikes (no decreases; +1 max; only one increments)
- `rules`: evolve balls/strikes deterministically from predicted Statcast `description`
- `constrained`: score rule-consistent transitions vs head transitions (outcome-aware models only)

Code: `src/baseball/simulate.py:1` (CLI: `src/baseball/cli/simulate.py:1`)

## Pitcher profiles (“policies”)

The `profile` command builds conditional policy tables from a trained model:
- predicted pitch-type mix per context bucket
- empirical pitch-type mix per bucket (ground truth)
- predicted vs empirical mean location per bucket

The repo supports multiple grouping keys, including batter-cluster conditioned variants (clustered from learned batter embeddings):
- `pitcher_count_prev_outcome`
- `pitcher_situation_prev_outcome`
- `pitcher_situation_batter_cluster_prev_outcome` (very granular; can be sparse)
- `pitcher_situation_batter_cluster` / `pitcher_situation_batter_cluster_prev` (less sparse; recommended starting point)

Code: `src/baseball/profile.py:1` (CLI: `src/baseball/cli/profile.py:1`)

## Example results (2023-only dataset)

One strong 2023-only run (teacher-forced valid), `final_state_mt_4665878`:
- pitch type top‑1 accuracy: **0.478**
- pitch type top‑3 accuracy: **0.897**
- pitch-type calibration (ECE, 15 bins): **0.041**
- location RMSE: **0.875 ft**

Empirical baselines (fit on train, evaluated on valid; same dataset):
- global pitch mix: **0.321** top‑1 acc
- pitcher-only: **0.418** top‑1 acc
- pitcher + count: **0.431** top‑1 acc
- pitcher + count + prev pitch type: **0.441** top‑1 acc

So the transformer clears a strong “what does this pitcher throw in this count?” baseline by ~**+3.7% absolute** top‑1 accuracy, while also improving location RMSE.

Slice highlights (valid split; from `report/slices.json`):
- accuracy is higher late innings: inning bucket `9+` is ~0.538 top‑1 vs ~0.468 in innings `1–3`
- accuracy rises with pitch number within the PA: pitch `4+` is ~0.500 top‑1 vs pitch `1` ~0.467

Open-loop rollouts on 50 held-out games (same model) show the expected drop from compounding state drift:
- rollout (heads): top‑1 **0.402**, top‑3 **0.830**, loc RMSE **0.898 ft**
- rollout (clamp count): top‑1 **0.403**, top‑3 **0.835**, loc RMSE **0.897 ft**

To reproduce on Greene:
```bash
export BASEBALL_ARTIFACT_ROOT="$VAST/baseball_pitch_model_desc_full2023"
python -m baseball report --run-id final_state_mt_4665878 --split valid
python -m baseball simulate --run-id final_state_mt_4665878 --split valid --mode rollout --count-mode clamp --max-games 50 --device cuda
```

## Limitations / what’s next

- The simulation does not model lineup/substitution decisions; it conditions on the real pitcher/batter IDs per pitch.
- Open-loop drift remains a major challenge; the repo measures it and includes stabilizers, but fully closing the gap is ongoing work.
- Batter clustering is currently a coarse proxy for “batter type”; richer batter representations (player-season embeddings, swing profiles, etc.) are natural next steps.

