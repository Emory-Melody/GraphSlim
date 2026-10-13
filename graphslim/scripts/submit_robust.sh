#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$graphslim_python" -m benchmark.reproduce robustness --slurm "$@"
