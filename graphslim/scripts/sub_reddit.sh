#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
rate="${1:?usage: sub_reddit.sh RATE [options]}"
shift
exec "$graphslim_python" -m graphslim.train_all "${graphslim_paths[@]}" -M gcdm --dis_metric l1 -D reddit -R "$rate" "$@"
