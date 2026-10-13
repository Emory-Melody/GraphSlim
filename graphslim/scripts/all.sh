#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: all.sh DATASET METHOD RATE ATTACK PTB_RATE [options]}"
method="${2:?missing method}"
rate="${3:?missing reduction rate}"
attack="${4:?missing attack}"
ptb_rate="${5:?missing perturbation rate}"
shift 5
exec "$graphslim_python" -m graphslim.train_all "${graphslim_paths[@]}" -D "$dataset" -M "$method" -R "$rate" -A "$attack" -P "$ptb_rate" "$@"
