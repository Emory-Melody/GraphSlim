#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: script_cross.sh DATASET RATE GPU [options]}"
rate="${2:?missing rate}"
gpu="${3:?missing GPU}"
shift 3
for seed in 0 1 2 3 4; do
    "$graphslim_python" -m benchmark.run_cross_arch "${graphslim_paths[@]}" -D "$dataset" -M gcond -R "$rate" -G "$gpu" -S "$seed" --run_eval 10 "$@"
done
