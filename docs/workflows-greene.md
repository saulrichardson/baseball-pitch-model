# Greene workflows

These are the cluster workflows used to run this project at scale.

**Key conventions**

- Clone repo at: `$VAST/baseball`
- Create venv at: `$VAST/baseball/.venv`
- Store all large artifacts under: `$VAST/...` via `BASEBALL_ARTIFACT_ROOT`

## Bootstrap

```bash
cd $VAST/baseball
bash scripts/greene/bootstrap_venv.sh
```

## Download + Prepare (scale)

Runs downloads (resumable by chunk) then prepares a sharded dataset without materializing the full season in RAM.

```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_transformer_scale,START=2023-03-30,END=2023-10-01,HISTORY_LEN=16,VALID_FRAC=0.1 scripts/greene/download_prepare_cpu.sbatch
```

## Train (GPU)

RTX8000 (older GPU, preferred):

```bash
cd $VAST/baseball
sbatch scripts/greene/train_rtx8000.sbatch
```

V100 (older GPU, smaller microbatches):

```bash
cd $VAST/baseball
sbatch scripts/greene/train_v100.sbatch
```

State-capable (for rollouts):

```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_transformer_full2023 scripts/greene/train_state_rtx8000.sbatch
```

## Report / Profile / Simulate

Report (baselines + calibration + slices):

```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_desc_full2023,RUN_ID=final_state_mt_4665878 scripts/greene/report_rtx8000.sbatch
```

Profile (pitcher policy tables):

```bash
cd $VAST/baseball
sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_desc_full2023,RUN_ID=final_state_mt_4665878,BY=pitcher_situation_prev_outcome,SPLIT=train,MIN_N=10 scripts/greene/profile_rtx8000.sbatch
```

Simulate (out-of-sample rollouts):

```bash
cd $VAST/baseball
python -m baseball simulate --run-id <RUN_ID> --split valid --mode rollout --count-mode clamp --max-games 50 --device cuda
```

## Full list of Slurm scripts

See: `scripts/greene/README.md`

