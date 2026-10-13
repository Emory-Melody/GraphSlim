#!/usr/bin/env bash
# Activate your environment before invoking a script. Override PYTHON if needed.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"
graphslim_python="${PYTHON:-python}"
graphslim_paths=(--load_path "${GRAPHSLIM_LOAD_PATH:-$repo_dir/data}"
                 --save_path "${GRAPHSLIM_SAVE_PATH:-$repo_dir/checkpoints}")
