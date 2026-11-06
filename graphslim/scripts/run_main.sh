#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$graphslim_python" -m benchmark.reproduce performance  "$@"
