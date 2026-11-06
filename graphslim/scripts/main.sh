#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
dataset="${1:?usage: main.sh DATASET METHOD RATE [options]}"
method="${2:?missing method}"
rate="${3:?missing reduction rate}"
shift 3
exec "$graphslim_python" -m graphslim.train_all "${graphslim_paths[@]}" -D "$dataset" -M "$method" -R "$rate" "$@"
