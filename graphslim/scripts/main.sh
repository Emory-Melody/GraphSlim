#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: main.sh DATASET METHOD RATE [options]}"
method="${2:?missing method}"
rate="${3:?missing reduction rate}"
shift 3
exec "$graphslim_python" -m graphslim.train_all "${graphslim_paths[@]}" -D "$dataset" -M "$method" -R "$rate" "$@"
