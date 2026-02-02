#!/usr/bin/env bash
#SBATCH --job-name=bb_smoke_state_mt
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000

set -euo pipefail

# NOTE: Do not `source /etc/profile.d/cluster.sh` inside Slurm jobs on Greene.
# It can terminate the shell early (no logs). The `module` function is already available.
module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-$VAST/baseball_pitch_model_desc_full2023}"
RUN_ID="${RUN_ID:-smoke_state_mt_${SLURM_JOB_ID}}"

cd "$REPO_DIR"
source .venv/bin/activate

export BASEBALL_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "REPO_DIR=$REPO_DIR"
echo "BASEBALL_ARTIFACT_ROOT=$BASEBALL_ARTIFACT_ROOT"
echo "RUN_ID=$RUN_ID"
python -V
nvidia-smi || true

# State-capable multi-task model (type + location + next-state + description) for open-loop rollouts.
python -m baseball train \
  --model transformer_mdn_state_mt \
  --run-id "$RUN_ID" \
  --epochs 3 \
  --batch-size 4096 \
  --grad-accum 2 \
  --lr 3e-4 \
  --seed 20260131 \
  --amp \
  --amp-dtype fp16 \
  --num-workers 2 \
  --prefetch-factor 2 \
  --streaming \
  --eval-max-batches 100

# Full eval on valid split for reporting.
python -m baseball eval --split valid --run-id "$RUN_ID" > "$ARTIFACT_ROOT/runs/$RUN_ID/eval_valid.json"

# Quick simulation sanity check (both modes) on a few games.
python -m baseball simulate --run-id "$RUN_ID" --split valid --mode replay --max-games 3 > "$ARTIFACT_ROOT/runs/$RUN_ID/sim_replay.json"
python -m baseball simulate --run-id "$RUN_ID" --split valid --mode rollout --max-games 3 > "$ARTIFACT_ROOT/runs/$RUN_ID/sim_rollout.json"

