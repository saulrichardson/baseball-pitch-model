#!/usr/bin/env bash
# Final state-capable training run (transformer + MDN + next-state heads).
#
# Produces:
#   - checkpoints/best.pt
#   - metrics.jsonl
#   - eval_valid.json (full eval)
#   - sim_replay_valid_500.json + sim_rollout_valid_500.json (full valid window; up to 500 games)
#   - exported bundle under models/exported/<RUN_ID>/
#
# Example:
#   sbatch --export=ALL,BASEBALL_ARTIFACT_ROOT=$VAST/baseball_pitch_model_state_full2023 scripts/greene/final_state_rtx8000.sbatch
#
#SBATCH --job-name=bb_final_state_rtx8000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000

set -euo pipefail

# NOTE: Do not `source /etc/profile.d/cluster.sh` inside Slurm jobs on Greene.
# It can terminate the shell early (no logs). The `module` function is already available.
module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-$VAST/baseball_pitch_model_state_full2023}"
RUN_ID="${RUN_ID:-final_state_${SLURM_JOB_ID}}"
MAX_GAMES="${MAX_GAMES:-500}"

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
echo "MAX_GAMES=$MAX_GAMES"
python -V
nvidia-smi || true

# Transformer config chosen to match the strongest pitch-only run, but with extra state heads.
python -m baseball train \
  --model transformer_mdn_state \
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

# Simulation on held-out games: replay (teacher-forced) + rollout (open-loop).
python -m baseball simulate \
  --run-id "$RUN_ID" \
  --split valid \
  --mode replay \
  --max-games "$MAX_GAMES" \
  --device cuda \
  --out "$ARTIFACT_ROOT/runs/$RUN_ID/sim_replay_valid_${MAX_GAMES}.json"

python -m baseball simulate \
  --run-id "$RUN_ID" \
  --split valid \
  --mode rollout \
  --max-games "$MAX_GAMES" \
  --device cuda \
  --out "$ARTIFACT_ROOT/runs/$RUN_ID/sim_rollout_valid_${MAX_GAMES}.json"

# Export best checkpoint for serving / offline inference.
python -m baseball export --run-id "$RUN_ID"
