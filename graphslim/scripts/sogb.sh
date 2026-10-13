#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$graphslim_python" -m graphslim.run_eval "${graphslim_paths[@]}" -M gcdm -D amazon --init clustering --agg -W "$@"
