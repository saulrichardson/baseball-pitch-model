#!/usr/bin/env bash
# Generate a report (metrics + baselines + calibration + slices) for a specific run.
#
# Environment variables:
#   - BASEBALL_ARTIFACT_ROOT (required)
#   - RUN_ID (required)
#   - SPLIT (optional): train|valid (default: valid)
#   - BASELINE_SPLIT (optional): train|valid (default: train)
#   - BATCH_SIZE (optional): model eval batch size (default: 4096)
#   - BASELINE_BATCH_SIZE (optional): baseline fit/eval batch size (default: 8192)
#   - BASELINE_ALPHA (optional): additive smoothing (default: 1.0)
#   - CALIBRATION_BINS (optional): ECE bins (default: 15)
#
# Example:
#   sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_desc_full2023,RUN_ID=final_state_mt_4665878 scripts/greene/report_rtx8000.sbatch
#
# Output:
#   $BASEBALL_ARTIFACT_ROOT/runs/$RUN_ID/report/{report.json,model_metrics.json,baselines.json,slices.json}
#
#SBATCH --job-name=bb_report_rtx8000
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
SPLIT="${SPLIT:-valid}"
BASELINE_SPLIT="${BASELINE_SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-8192}"
BASELINE_ALPHA="${BASELINE_ALPHA:-1.0}"
CALIBRATION_BINS="${CALIBRATION_BINS:-15}"

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
echo "SPLIT=$SPLIT BASELINE_SPLIT=$BASELINE_SPLIT"
echo "BATCH_SIZE=$BATCH_SIZE BASELINE_BATCH_SIZE=$BASELINE_BATCH_SIZE BASELINE_ALPHA=$BASELINE_ALPHA CALIBRATION_BINS=$CALIBRATION_BINS"
python -V
nvidia-smi || true

python -m baseball report \
  --run-id "$RUN_ID" \
  --split "$SPLIT" \
  --baseline-split "$BASELINE_SPLIT" \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --baseline-batch-size "$BASELINE_BATCH_SIZE" \
  --baseline-alpha "$BASELINE_ALPHA" \
  --calibration-bins "$CALIBRATION_BINS"

