# Greene workflows

All scripts assume:
- the repo is cloned at `$VAST/baseball`
- a python venv exists at `$VAST/baseball/.venv`
- artifacts live under `$VAST` (override with `$BASEBALL_ARTIFACT_ROOT`)

Prepared datasets are written as sharded parquet directories under:
- `$BASEBALL_ARTIFACT_ROOT/data/prepared/train/*.parquet`
- `$BASEBALL_ARTIFACT_ROOT/data/prepared/valid/*.parquet`

## Bootstrap

```bash
cd $VAST/baseball
bash scripts/greene/bootstrap_venv.sh
```

## Download + Prepare (scale)

Runs downloads (resumable by chunk) then prepares a sharded dataset without
materializing the full season in RAM.

```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_transformer_scale,START=2023-03-30,END=2023-10-01,HISTORY_LEN=16,VALID_FRAC=0.1 scripts/greene/download_prepare_cpu.sbatch
```

### Vocab / OOV tuning

If you want fewer OOV pitcher/batter IDs in validation (at the cost of bigger vocabs), pass lower minimums:

```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_desc_full2023,START=2023-03-30,END=2023-10-01,HISTORY_LEN=16,VALID_FRAC=0.1,MIN_PITCHER_COUNT=1,MIN_BATTER_COUNT=1 scripts/greene/download_prepare_cpu.sbatch
```

## Train

CPU:
```bash
cd $VAST/baseball
sbatch --partition=cs scripts/greene/train_cpu.sbatch
```

RTX8000 (older GPU, preferred for throughput):
```bash
cd $VAST/baseball
sbatch scripts/greene/train_rtx8000.sbatch
```

V100 (older GPU, smaller microbatches):
```bash
cd $VAST/baseball
sbatch scripts/greene/train_v100.sbatch
```

State-capable (for open-loop rollouts / simulation):
```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_transformer_full2023 scripts/greene/train_state_rtx8000.sbatch
```

Multi-task (predicts pitch `description_id` in addition to pitch type + location):
```bash
cd $VAST/baseball
sbatch scripts/greene/smoke_mdn_mt_rtx8000.sbatch
```

Multi-task + state-capable (for open-loop rollouts with predicted outcome tokens):
```bash
cd $VAST/baseball
sbatch scripts/greene/smoke_state_mt_rtx8000.sbatch
```

GPU (A100 partition may require access):
```bash
cd $VAST/baseball
sbatch scripts/greene/train_a100.sbatch
```

## Sweep (job arrays)

RTX8000:
```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_transformer_scale scripts/greene/sweep_rtx8000.sbatch
```

V100:
```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_transformer_scale scripts/greene/sweep_v100.sbatch
```

## Evaluate

```bash
cd $VAST/baseball
sbatch --partition=cs scripts/greene/eval_cpu.sbatch
```

## Simulate (out-of-sample games)

Requires a `transformer_mdn_state` run (state head).

```bash
cd $VAST/baseball
python -m baseball simulate --run-id <RUN_ID> --split valid --mode replay --max-games 5
python -m baseball simulate --run-id <RUN_ID> --split valid --mode rollout --max-games 5
```

If you train `transformer_mdn_state_mt`, rollouts will also evolve within-PA outcome history (`hist_desc`) using
predicted `description_id`, instead of leaving it unknown.

## Report (baselines + calibration + slices)

Generates a writeup-ready evaluation under:
`$BASEBALL_ARTIFACT_ROOT/runs/<RUN_ID>/report/`

```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_desc_full2023,RUN_ID=final_state_mt_4665878 scripts/greene/report_rtx8000.sbatch
```

## Profile (pitcher policies)

Builds per-pitcher conditional pitch-mix + location tables.

```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_desc_full2023,RUN_ID=final_state_mt_4665878,BY=pitcher_situation_prev_outcome,SPLIT=train,MIN_N=10 scripts/greene/profile_rtx8000.sbatch
```

## Serve

```bash
cd $VAST/baseball
sbatch --partition=cs scripts/greene/serve_cpu.sbatch
```
