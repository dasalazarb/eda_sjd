#!/usr/bin/env bash
#SBATCH --job-name=sjd_harmonize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
umask 077
repo="${SLURM_SUBMIT_DIR:?Submit from the eda_sjd repository root}"
module_dir="$repo/src/semantic_harmonization"
environment="${SJD_SEMANTIC_VENV:-$repo/.venv-semantic}"
log_dir="$repo/work/semantic_harmonization"
mkdir -p "$log_dir"
exec >"$log_dir/job-${SLURM_JOB_ID}.log" 2>&1
source "$environment/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name())'
exec python "$module_dir/run_pipeline.py"
