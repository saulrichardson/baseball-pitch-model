#!/usr/bin/env bash
#SBATCH --job-name=bb_final_rtx8000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000

set -euo pipefail

# NOTE: Do not `source /etc/profile.d/cluster.sh` inside Slurm jobs on Greene.
# It can terminate the shell early (no logs). The `module` function is already available.
module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-$VAST/baseball_pitch_model_transformer_full2023}"
RUN_ID="${RUN_ID:-final_${SLURM_JOB_ID}}"

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

# Chosen from the RTX8000 sweep as best (ce + 0.3*loc_nll) on full-season 2023 valid.
# Effective batch = 4096 * 2 = 8192.
python -m baseball train \
  --model transformer_mdn \
  --run-id "$RUN_ID" \
  --epochs 25 \
  --batch-size 4096 \
  --grad-accum 2 \
  --lr 2e-4 \
  --seed 20260129 \
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
  --dropout 0.1

# Full eval on valid split for reporting.
python -m baseball eval --split valid --run-id "$RUN_ID" > "$ARTIFACT_ROOT/runs/$RUN_ID/eval_valid.json"

# Export best checkpoint for serving.
python -m baseball export --run-id "$RUN_ID"

