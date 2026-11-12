#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
for arg in "$@"; do
  [[ "$arg" != --resources ]] || exec cards status
 done
if [[ "${RDO_PPU_CONTAINER:-0}" != 1 && -e /dev/alixpu && "${RDO_PLATFORM:-auto}" != cpu ]]; then
  exec bash "$PWD/scripts/launch_ppu.sh" dim "$@"
fi
repo_root="$PWD"
python_bin="${RDO_PYTHON:-$repo_root/.venv/bin/python}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd geometry-of-refusal/refusal_direction
exec "$python_bin" -u -m pipeline.run_pipeline "$@"
