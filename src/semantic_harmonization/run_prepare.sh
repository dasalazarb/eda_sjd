#!/usr/bin/env bash
set -euo pipefail
umask 077
module_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module_dir/../.." && pwd -P)"
environment="${SJD_SEMANTIC_VENV:-$repo/.venv-semantic}"
source "$environment/bin/activate"
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
exec python "$module_dir/00_prepare_resources.py"
