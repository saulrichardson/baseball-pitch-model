#!/usr/bin/env bash
# Simulate held-out games (replay + rollout) using a state-capable model.
#
# Outputs JSON summaries into:
#   $BASEBALL_ARTIFACT_ROOT/runs/$RUN_ID/
#
# Environment variables:
#   - BASEBALL_ARTIFACT_ROOT (required): artifact root containing data/prepared and runs/
#   - RUN_ID (required): run directory name under runs/ (must contain checkpoints/best.pt)
#   - MAX_GAMES (optional): number of games to simulate (default: 500)
#   - COUNT_MODE (optional): heads|clamp|rules|constrained for rollout count updates (default: heads)
#
# Notes:
#   - Uses RTX8000 (older GPU) + fp16-friendly configs by default.
#   - Keeps CPU RAM request low; polars scans parquet lazily per-game.
#
# Example:
#   sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_state_full2023,RUN_ID=state_full2023_rtx,MAX_GAMES=500 scripts/greene/simulate_state_rtx8000.sbatch
#
#SBATCH --job-name=bb_sim_state_rtx8000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000

set -euo pipefail

# NOTE: Do not `source /etc/profile.d/cluster.sh` inside Slurm jobs on Greene.
# It can terminate the shell early (no logs). The `module` function is already available.
module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-}"
RUN_ID="${RUN_ID:-}"
MAX_GAMES="${MAX_GAMES:-500}"
COUNT_MODE="${COUNT_MODE:-heads}"

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

RUN_DIR="$BASEBALL_ARTIFACT_ROOT/runs/$RUN_ID"
CKPT_PATH="$RUN_DIR/checkpoints/best.pt"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: missing checkpoint: $CKPT_PATH" 1>&2
  exit 3
fi

echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "REPO_DIR=$REPO_DIR"
echo "BASEBALL_ARTIFACT_ROOT=$BASEBALL_ARTIFACT_ROOT"
echo "RUN_ID=$RUN_ID"
echo "MAX_GAMES=$MAX_GAMES"
echo "COUNT_MODE=$COUNT_MODE"
python -V
nvidia-smi || true

python -m baseball simulate \
  --run-id "$RUN_ID" \
  --split valid \
  --mode replay \
  --count-mode "$COUNT_MODE" \
  --max-games "$MAX_GAMES" \
  --device cuda \
  --out "$RUN_DIR/sim_replay_valid_${MAX_GAMES}.json"

python -m baseball simulate \
  --run-id "$RUN_ID" \
  --split valid \
  --mode rollout \
  --count-mode "$COUNT_MODE" \
  --max-games "$MAX_GAMES" \
  --device cuda \
  --out "$RUN_DIR/sim_rollout_${COUNT_MODE}_valid_${MAX_GAMES}.json"
