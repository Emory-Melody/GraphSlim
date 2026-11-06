#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
rate="${1:?usage: sub_reddit.sh RATE [options]}"
shift
exec "$graphslim_python" -m graphslim.train_all "${graphslim_paths[@]}" -M gcdm --dis_metric l1 -D reddit -R "$rate" "$@"
