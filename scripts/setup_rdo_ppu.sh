#!/bin/bash
# Run with the selected image's Python, never the host NVIDIA venv.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "PPU setup must run inside a Slurm labs-pool job." >&2; exit 1; }
cd "$(dirname "$0")/.."
venv_dir="${RDO_PPU_VENV:-$PWD/.venv-ppu-sail21}"
python3 -m venv --system-site-packages "$venv_dir"
"$venv_dir/bin/python" - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys
# These wheels have vendor ABI dependencies and must remain in the image.
protected = ['torch', 'torchvision', 'torchaudio', 'numpy', 'triton', 'bitsandbytes']
lines = []
for name in protected:
    try:
        lines.append(f'{name}=={version(name)}')
    except PackageNotFoundError:
        pass
Path(sys.prefix, 'vendor-constraints.txt').write_text('\n'.join(lines) + '\n')
PY
"$venv_dir/bin/python" -m pip install -c "$venv_dir/vendor-constraints.txt" -r requirements-ppu.txt
"$venv_dir/bin/python" - <<'PY'
import torch
from pathlib import Path
import sys
# The runtime must still come from the image rather than a venv wheel.
if Path(torch.__file__).is_relative_to(Path(sys.prefix)):
    raise RuntimeError('Vendor torch was shadowed in the PPU venv; rebuild this venv.')
import nnsight, transformers
import numpy as np, pyarrow as pa, datasets
assert np.array_equal(pa.array(np.array([0, 1])).to_numpy(), [0, 1])
assert datasets.Dataset.from_dict({'answer': [0]})[0]['answer'] == 0
print(f'[ppu] Data dependencies: numpy={np.__version__} pyarrow={pa.__version__}')
print(f'[ppu] Application environment ready: torch={torch.__version__}, transformers={transformers.__version__}')
PY
sha256sum requirements-ppu.txt scripts/setup_rdo_ppu.sh > "$venv_dir/.rdo-requirements.sha256"
