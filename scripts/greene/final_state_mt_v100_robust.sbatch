#!/usr/bin/env bash
# V100 variant of the robust state MT training job.
#
# Uses:
#   - smaller microbatch (VRAM)
#   - gradient checkpointing
#   - state corruption on balls/strikes
#
#SBATCH --job-name=bb_final_state_mt_v100_robust
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=v100

set -euo pipefail

module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-$VAST/baseball_pitch_model_desc_full2023}"
RUN_ID="${RUN_ID:-final_state_mt_v100_robust_${SLURM_JOB_ID}}"

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

python -m baseball train \
  --model transformer_mdn_state_mt \
  --run-id "$RUN_ID" \
  --epochs 15 \
  --batch-size 2048 \
  --grad-accum 4 \
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

python -m baseball eval --split valid --run-id "$RUN_ID" > "$ARTIFACT_ROOT/runs/$RUN_ID/eval_valid.json"
python -m baseball simulate --run-id "$RUN_ID" --split valid --mode replay --max-games 20 > "$ARTIFACT_ROOT/runs/$RUN_ID/sim_replay.json"
python -m baseball simulate --run-id "$RUN_ID" --split valid --mode rollout --count-mode heads --max-games 20 > "$ARTIFACT_ROOT/runs/$RUN_ID/sim_rollout_heads.json"
python -m baseball simulate --run-id "$RUN_ID" --split valid --mode rollout --count-mode constrained --max-games 20 > "$ARTIFACT_ROOT/runs/$RUN_ID/sim_rollout_constrained.json"
python -m baseball export --run-id "$RUN_ID"

