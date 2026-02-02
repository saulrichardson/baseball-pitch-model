#!/usr/bin/env bash
#SBATCH --job-name=bb_train_cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

set -euo pipefail

# NOTE: Do not `source /etc/profile.d/cluster.sh` inside Slurm jobs on Greene.
# It can terminate the shell early (no logs). The `module` function is already available.
module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-$VAST/baseball_pitch_model}"
RUN_ID="${RUN_ID:-}"

cd "$REPO_DIR"
source .venv/bin/activate

export BASEBALL_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "REPO_DIR=$REPO_DIR"
echo "BASEBALL_ARTIFACT_ROOT=$BASEBALL_ARTIFACT_ROOT"
python -V

RUN_ARGS=()
if [[ -n "$RUN_ID" ]]; then
  RUN_ARGS+=(--run-id "$RUN_ID")
fi

python -m baseball train \
  --model transformer_mdn \
  "${RUN_ARGS[@]}" \
  --epochs 10 \
  --batch-size 4096 \
  --grad-accum 1 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --streaming \
  --eval-max-batches 200
