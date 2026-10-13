#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
rate="${1:?usage: sub_eval.sh RATE [options]}"
shift
exec "$graphslim_python" -m graphslim.run_eval "${graphslim_paths[@]}" -M gcdm -D yelp -R "$rate" -W --run_eval 10 --eval_epochs 1000 "$@"
