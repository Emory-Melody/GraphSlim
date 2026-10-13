#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../../scripts/common.sh"
exec "$graphslim_python" -m benchmark.reproduce performance "$@"
