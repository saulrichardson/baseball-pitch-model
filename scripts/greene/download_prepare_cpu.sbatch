#!/usr/bin/env bash
#SBATCH --job-name=bb_dl_prep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cs

set -euo pipefail

# NOTE: Do not `source /etc/profile.d/cluster.sh` inside Slurm jobs on Greene.
# It can terminate the shell early (no logs). The `module` function is already available.
module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-$VAST/baseball_pitch_model_transformer_scale}"

# Dataset + split controls (override via sbatch --export=ALL,START=...,END=...).
START="${START:-2023-03-30}"
END="${END:-2023-10-01}"
CHUNK_DAYS="${CHUNK_DAYS:-7}"
HISTORY_LEN="${HISTORY_LEN:-16}"
VALID_FRAC="${VALID_FRAC:-0.1}"
MIN_PITCH_TYPE_COUNT="${MIN_PITCH_TYPE_COUNT:-50}"
MIN_DESCRIPTION_COUNT="${MIN_DESCRIPTION_COUNT:-25}"
MIN_PITCHER_COUNT="${MIN_PITCHER_COUNT:-50}"
MIN_BATTER_COUNT="${MIN_BATTER_COUNT:-50}"

cd "$REPO_DIR"
source .venv/bin/activate

export BASEBALL_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "REPO_DIR=$REPO_DIR"
echo "BASEBALL_ARTIFACT_ROOT=$BASEBALL_ARTIFACT_ROOT"
echo "START=$START END=$END CHUNK_DAYS=$CHUNK_DAYS HISTORY_LEN=$HISTORY_LEN VALID_FRAC=$VALID_FRAC"
echo "MIN_PITCH_TYPE_COUNT=$MIN_PITCH_TYPE_COUNT MIN_DESCRIPTION_COUNT=$MIN_DESCRIPTION_COUNT MIN_PITCHER_COUNT=$MIN_PITCHER_COUNT MIN_BATTER_COUNT=$MIN_BATTER_COUNT"
python -V

python -m baseball download \
  --start "$START" \
  --end "$END" \
  --chunk-days "$CHUNK_DAYS"

python -m baseball prepare \
  --start "$START" \
  --end "$END" \
  --history-len "$HISTORY_LEN" \
  --valid-frac "$VALID_FRAC" \
  --min-pitch-type-count "$MIN_PITCH_TYPE_COUNT" \
  --min-description-count "$MIN_DESCRIPTION_COUNT" \
  --min-pitcher-count "$MIN_PITCHER_COUNT" \
  --min-batter-count "$MIN_BATTER_COUNT" \
  --overwrite
