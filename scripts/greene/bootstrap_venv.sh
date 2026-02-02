#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VAST:-}" ]]; then
  echo "VAST is not set. On Greene, VAST should point to /vast/<netid>." >&2
  exit 1
fi

REPO_DIR="${1:-$VAST/baseball}"
VENV_DIR="${2:-$REPO_DIR/.venv}"

cd "$REPO_DIR"

# NOTE: Do not `source /etc/profile.d/cluster.sh` on Greene in batch contexts; it can terminate the shell.
# The `module` function is already available.
module load anaconda3/2024.02

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip
python -m pip install -e .

echo "OK: venv ready at $VENV_DIR"
