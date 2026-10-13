#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: nas.sh DATASET METHOD RATE [options]}"
method="${2:?missing method}"
rate="${3:?missing reduction rate}"
shift 3
exec "$graphslim_python" -m benchmark.run_nas "${graphslim_paths[@]}" -D "$dataset" -M "$method" -R "$rate" "$@"
