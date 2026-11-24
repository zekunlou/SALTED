#!/usr/bin/env bash
# Run from a SALTED checkout in an environment with its test dependencies.
set -euo pipefail

: "${SALTED_DATASETS_PATH:?Set SALTED_DATASETS_PATH to the SALTED-datasets checkout}"
datasets=$(realpath -- "$SALTED_DATASETS_PATH")
cd "$(dirname "$0")/.."
python=${PYTHON:-python}
threads=${SALTED_BENCH_THREADS:-1}
export OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OMP_NUM_THREADS="$threads" NUMBA_NUM_THREADS="$threads"

mkdir -p playground/runs
run=$(mktemp -d "$PWD/playground/runs/aims-density.XXXXXXXX")
echo "Run directory: $run"
export NUMBA_CACHE_DIR="$run/numba-cache"
{
    echo "SALTED commit: $(git rev-parse HEAD)"
    git status --short
    echo "Datasets: $datasets"
    git -C "$datasets" rev-parse HEAD
    git -C "$datasets" status --short
    echo "Threads: $threads"
    uname -srmo
    "$python" -VV
    "$python" -m pip freeze --local
} > "$run/environment.txt"

/usr/bin/time -f 'pipeline_wall_seconds=%e\npipeline_peak_rss_kib=%M' \
    -o "$run/resources.txt" \
    "$python" -m pytest tests/integration/test_pipeline.py -m aims_density \
    --require-datasets --datasets-path "$datasets" --basetemp "$run/work" -q -s \
    2>&1 | tee "$run/pytest.log"

grep -Eq '(^|[[:space:]])1 passed([,[:space:]]|$)' "$run/pytest.log"
cat "$run/resources.txt"
grep -E '^ +((hessian_matrix|% RMSE):)' "$run/pytest.log"
