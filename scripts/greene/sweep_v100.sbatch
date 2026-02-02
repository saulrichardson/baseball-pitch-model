#!/usr/bin/env bash
#SBATCH --job-name=bb_sweep_v100
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=v100
#SBATCH --array=0-5

set -euo pipefail

# NOTE: Do not `source /etc/profile.d/cluster.sh` inside Slurm jobs on Greene.
# It can terminate the shell early (no logs). The `module` function is already available.
module load anaconda3/2024.02

REPO_DIR="${REPO_DIR:-$VAST/baseball}"
ARTIFACT_ROOT="${BASEBALL_ARTIFACT_ROOT:-$VAST/baseball_pitch_model_transformer_scale}"

cd "$REPO_DIR"
source .venv/bin/activate

export BASEBALL_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

IDX="${SLURM_ARRAY_TASK_ID}"
RUN_ID="sweep_${SLURM_JOB_ID}_${IDX}"

echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "REPO_DIR=$REPO_DIR"
echo "BASEBALL_ARTIFACT_ROOT=$BASEBALL_ARTIFACT_ROOT"
echo "IDX=$IDX RUN_ID=$RUN_ID"
python -V
nvidia-smi || true

# V100 tends to have less VRAM than RTX8000, so keep d_model smaller.
D_MODEL=(128 192 256 256 192 256)
LAYERS=(4   4   4   6   6   6)
NHEAD=(8    8   8   8   8   8)
MDN_K=(8    8   8   8   12  16)
LR=(3e-4 3e-4 2e-4 2e-4 2e-4 1e-4)
DROPOUT=(0.1 0.1 0.1 0.1 0.1 0.1)

echo "CONFIG d_model=${D_MODEL[$IDX]} layers=${LAYERS[$IDX]} nhead=${NHEAD[$IDX]} mdn_k=${MDN_K[$IDX]} lr=${LR[$IDX]} dropout=${DROPOUT[$IDX]}"

python -m baseball train \
  --model transformer_mdn \
  --run-id "$RUN_ID" \
  --epochs 6 \
  --batch-size 2048 \
  --grad-accum 4 \
  --lr "${LR[$IDX]}" \
  --seed "$((1337 + IDX))" \
  --amp \
  --amp-dtype fp16 \
  --num-workers 2 \
  --prefetch-factor 2 \
  --streaming \
  --eval-max-batches 100 \
  --d-model "${D_MODEL[$IDX]}" \
  --layers "${LAYERS[$IDX]}" \
  --nhead "${NHEAD[$IDX]}" \
  --mdn-components "${MDN_K[$IDX]}" \
  --dropout "${DROPOUT[$IDX]}"

python -m baseball eval --split valid --run-id "$RUN_ID" > "$ARTIFACT_ROOT/runs/$RUN_ID/eval_valid.json"
