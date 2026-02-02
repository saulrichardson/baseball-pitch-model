#!/usr/bin/env bash
# Build pitcher profiles (policy tables) from a trained model.
#
# Environment variables:
#   - BASEBALL_ARTIFACT_ROOT (required)
#   - RUN_ID (required)
#   - BY (optional): profile aggregation key (default: pitcher_situation_prev_outcome)
#   - SPLIT (optional): train|valid (default: valid)
#   - BATCH_SIZE (optional): streaming batch size (default: 8192)
#   - MIN_N (optional): minimum examples per group (default: 10)
#   - BATTER_CLUSTERS (optional): KMeans K (only for batter_cluster by; default: 32)
#
# Example:
#   sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_desc_full2023,RUN_ID=final_state_mt_4665878,BY=pitcher_situation_prev_outcome scripts/greene/profile_rtx8000.sbatch
#
#SBATCH --job-name=bb_profile_rtx8000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000

set -euo pipefail

module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-}"
RUN_ID="${RUN_ID:-}"
BY="${BY:-pitcher_situation_prev_outcome}"
SPLIT="${SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
MIN_N="${MIN_N:-10}"
BATTER_CLUSTERS="${BATTER_CLUSTERS:-32}"

if [[ -z "$ARTIFACT_ROOT" ]]; then
  echo "ERROR: BASEBALL_ARTIFACT_ROOT is required." 1>&2
  exit 2
fi
if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: RUN_ID is required." 1>&2
  exit 2
fi

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
echo "BY=$BY SPLIT=$SPLIT BATCH_SIZE=$BATCH_SIZE MIN_N=$MIN_N BATTER_CLUSTERS=$BATTER_CLUSTERS"
python -V
nvidia-smi || true

python -m baseball profile \
  --run-id "$RUN_ID" \
  --split "$SPLIT" \
  --by "$BY" \
  --batch-size "$BATCH_SIZE" \
  --min-n "$MIN_N" \
  --batter-clusters "$BATTER_CLUSTERS" \
  --device cuda
