#!/usr/bin/env bash
# Track4 Risk-Gate V1 shell entry.
# This file is a compatibility wrapper; code/run.py is the canonical entry.
# It never checks or requires the 27B teacher/LoRA files.
set -euo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

if [ -f /etc/profile.d/dockerenv.sh ]; then
  # Provides the PPU/torch environment on the contest image.
  # shellcheck disable=SC1091
  source /etc/profile.d/dockerenv.sh
fi

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV HF_HOME HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE
while IFS= read -r task_key; do
  if [[ "$task_key" == TRACK4_* ]]; then unset "$task_key"; fi
done < <(compgen -e)
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0

required="import torch,transformers,yaml,sklearn,numpy,PIL"
python_bin=""
for candidate in \
  "${TRACK4_PYTHON:-}" \
  "~/track4_current/venv/bin/python" \
  "/usr/local/bin/python3" \
  "$(command -v python3 || true)"; do
  [ -n "$candidate" ] || continue
  [ -x "$candidate" ] || continue
  if "$candidate" -c "$required" >/dev/null 2>&1; then
    python_bin="$candidate"
    break
  fi
done

if [ -z "$python_bin" ]; then
  echo "No Python with torch/transformers/yaml/sklearn/numpy/PIL found. Set TRACK4_PYTHON." >&2
  exit 3
fi

echo "[run.sh] python=$python_bin" >&2
exec "$python_bin" "$release_root/run.py" "$@"
