#!/bin/bash
# Shared scheduler/container boundary for RDO and geometry-of-refusal DIM.
set -euo pipefail
if [[ -n "${RDO_REPO_ROOT:-}" ]]; then
  repo_root="$RDO_REPO_ROOT"
elif [[ -f "$(dirname "${BASH_SOURCE[0]}")/../requirements-ppu.txt" ]]; then
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/requirements-ppu.txt" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  echo 'Cannot find repository; set RDO_REPO_ROOT.' >&2; exit 1
fi
cd "$repo_root"
export RDO_REPO_ROOT="$PWD"
workload="${1:?Expected rdo or dim}"; shift
case "$workload" in rdo|dim) ;; *) echo 'Expected rdo or dim' >&2; exit 2 ;; esac
image="${RDO_PPU_IMAGE:-/bmcp_lvm_fs/apptainer/sif/inference-26.04-v2.1.0-vllm0.20.1-torch2.10-cu130-20260608.sif}"
venv_dir="${RDO_PPU_VENV:-$repo_root/.venv-ppu-sail21}"
profile="${RDO_PPU_PROFILE:-whole}"
case "$profile" in whole|half|auto) ;; *) echo 'RDO_PPU_PROFILE must be whole, half or auto' >&2; exit 2 ;; esac
[[ -f "$image" ]] || { echo "Image missing: $image" >&2; exit 1; }
check_only=0
for arg in "$@"; do
  case "$arg" in --resources) exec cards status ;; --check_environment|--smoke-test|--setup-only) check_only=1 ;; esac
done
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  resources=(--partition=labs-pool --qos=pool-batch --nodes=1 --ntasks=1
    --cpus-per-task="${RDO_PPU_CPUS:-8}" --mem="${RDO_PPU_RAM:-128G}" --job-name="$workload-ppu")
  gres=(--gres=gpu:1)
  [[ "$profile" != whole ]] || gres+=(--comment=ppu=whole)
  [[ "$profile" != half ]] || gres=(--gres=shard:1)
  if ((check_only)); then
    echo "[ppu] Short $workload check/setup in labs-pool (profile=$profile)."
    exec srun "${resources[@]}" "${gres[@]}" --time=00:30:00 --immediate=30 \
      bash "$repo_root/scripts/launch_ppu.sh" "$workload" "$@"
  fi
  mkdir -p "$repo_root/results/rdo_refusal/slurm"
  echo "[ppu] Submitting $workload to labs-pool, profile=$profile; device assignment is managed by Slurm."
  exec sbatch --chdir="$repo_root" --export=ALL "${resources[@]}" "${gres[@]}" \
    --time="${RDO_PPU_TIME:-24:00:00}" --open-mode=append \
    --output="$repo_root/results/rdo_refusal/slurm/$workload-ppu-%j.log" \
    "$repo_root/scripts/launch_ppu.sh" "$workload" "$@"
fi
[[ -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}${APPTAINERENV_CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo 'Slurm job has no PPU allocation.' >&2; exit 1;
}
# Preserve the TaskProlog-provided device mask, particularly the MIG UUID.
export APPTAINERENV_RDO_PPU_CONTAINER=1 APPTAINERENV_RDO_PLATFORM=ppu
export APPTAINERENV_RDO_DEVICE=cuda APPTAINERENV_RDO_PYTHON="$venv_dir/bin/python"
export APPTAINERENV_RDO_PPU_VENV="$venv_dir"
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Forward application settings, without placing secret values in command arguments.
while IFS= read -r key; do
  case "$key" in
    HF_TOKEN|HF_HOME|HF_ENDPOINT|HUGGINGFACE_CACHE_DIR|OPENAI_API_KEY|ANTHROPIC_API_KEY|MODEL_NAME|DIRECTION_MODE|NUM_OPT_LAYERS|K_PROJ|ORTHOGONALIZE|TRAIN_DIAGNOSTICS_GAP|RESULT_ROOT|MAX_ITERS|*_LAMBDA|MMLU_*|LLAMAGUARD_*|GUARD_*|PPL_*|ARC_*|GSM8K_*|DIM_BATCH_SIZE|EVAL_ACTIVATION_PCA|ACTIVATION_SAVE_GAP|KEEP_ACTIVATION_PCA_DUMPS)
      export "APPTAINERENV_$key=${!key}" ;;
  esac
done < <(compgen -e)
if [[ ! -f "$venv_dir/pyvenv.cfg" || ! -f "$venv_dir/.rdo-requirements.sha256" ]] ||
   ! sha256sum --status -c "$venv_dir/.rdo-requirements.sha256"; then
  apptainer exec --cleanenv "$image" bash "$repo_root/scripts/setup_rdo_ppu.sh"
fi
for arg in "$@"; do
  [[ "$arg" != --setup-only ]] || exit 0
  if [[ "$arg" == --smoke-test ]]; then
    exec apptainer exec --cleanenv "$image" "$venv_dir/bin/python" "$repo_root/scripts/smoke_ppu.py" "$workload"
  fi
done
entry="$repo_root/scripts/run_rdo_refusal.sh"
[[ "$workload" != dim ]] || entry="$repo_root/scripts/run_dim_pipeline.sh"
exec apptainer exec --cleanenv "$image" bash "$entry" "$@"
