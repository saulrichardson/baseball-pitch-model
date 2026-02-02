#!/usr/bin/env bash
# Train a state-capable multi-task transformer with rollout-robustness knobs enabled.
#
# Adds:
#   - --grad-checkpointing (GPU memory saver for older GPUs)
#   - --state-corrupt-p (train-time corruption of balls/strikes to improve open-loop stability)
#
# Outputs:
#   $BASEBALL_ARTIFACT_ROOT/runs/$RUN_ID/
#
# Example:
#   sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_desc_full2023 scripts/greene/final_state_mt_rtx8000_robust.sbatch
#
#SBATCH --job-name=bb_final_state_mt_robust
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000

set -euo pipefail

# NOTE: Do not `source /etc/profile.d/cluster.sh` inside Slurm jobs on Greene.
# It can terminate the shell early (no logs). The `module` function is already available.
module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-$VAST/baseball_pitch_model_desc_full2023}"
RUN_ID="${RUN_ID:-final_state_mt_robust_${SLURM_JOB_ID}}"

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

# State-capable multi-task model:
# - pitch type + location (MDN)
# - next-state prediction (for open-loop rollout)
# - pitch description prediction (to evolve within-PA outcome history during rollout)
python -m baseball train \
  --model transformer_mdn_state_mt \
  --run-id "$RUN_ID" \
  --epochs 15 \
  --batch-size 4096 \
  --grad-accum 2 \
  --lr 2e-4 \
  --seed 20260131 \
  --amp \
  --amp-dtype fp16 \
  --num-workers 2 \
  --prefetch-factor 2 \
  --streaming \
  --eval-max-batches 200 \
  --d-model 256 \
  --layers 6 \
  --nhead 8 \
  --mdn-components 8 \
  --dropout 0.1 \
  --desc-weight 0.05 \
  --state-weight 0.2 \
  --loc-weight 0.3 \
  --grad-checkpointing \
  --state-corrupt-p 0.05 \
  --state-corrupt-max-delta 1

# Full eval on valid split for reporting.
python -m baseball eval --split valid --run-id "$RUN_ID" > "$ARTIFACT_ROOT/runs/$RUN_ID/eval_valid.json"

# Simulate held-out games (replay vs rollout) to measure drift.
python -m baseball simulate --run-id "$RUN_ID" --split valid --mode replay --max-games 50 > "$ARTIFACT_ROOT/runs/$RUN_ID/sim_replay.json"

# Rollout variants:
# - heads: trust the state heads for balls/strikes/pa_end.
# - constrained: choose a consistent transition using both description + state heads.
python -m baseball simulate --run-id "$RUN_ID" --split valid --mode rollout --count-mode heads --max-games 50 > "$ARTIFACT_ROOT/runs/$RUN_ID/sim_rollout_heads.json"
python -m baseball simulate --run-id "$RUN_ID" --split valid --mode rollout --count-mode constrained --max-games 50 > "$ARTIFACT_ROOT/runs/$RUN_ID/sim_rollout_constrained.json"

# Export for serving.
python -m baseball export --run-id "$RUN_ID"

