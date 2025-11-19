#!/bin/sh
set -e

TRIALS=${1:-50}

echo "=== Configuring CMake (if needed) ==="
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

cmake -DCMAKE_BUILD_TYPE=Release .. >/dev/null

echo "=== Building C++ benchmark (CMake) ==="
cmake --build . --config Release -j8

echo "=== Running C++ benchmark with $TRIALS trials ==="
./output/benchmark_stats2 "$TRIALS"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating Python venv ==="
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

echo "=== Upgrading pip and installing Python deps ==="
pip install --upgrade pip >/dev/null
pip install -r ../requirements.txt >/dev/null 2>&1 || pip install pandas numpy matplotlib >/dev/null

echo "=== Rendering plots and LaTeX tables ==="
python ../render_benchmarks.py

echo "=== Done. Results in figs/ and tables/. ==="