#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: transferbility.sh DATASET METHOD [options]}"
method="${2:?missing method}"
shift 2
exec "$graphslim_python" -m benchmark.run_cross_arch "${graphslim_paths[@]}" -D "$dataset" -M "$method" "$@"
