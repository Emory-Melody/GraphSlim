#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
rate="${1:?usage: sub_eval.sh RATE [options]}"
shift
exec "$graphslim_python" -m graphslim.run_eval "${graphslim_paths[@]}" -M gcdm -D yelp -R "$rate" -W --run_eval 10 --eval_epochs 1000 "$@"
