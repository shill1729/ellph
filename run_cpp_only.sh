#!/bin/sh
set -e

TRIALS=${1:-50}

if [ ! -d "build" ]; then
    echo "CMake has not been configured. Run cmake first."
    exit 1
fi

cd build
cmake --build . --config Release -j8

./output/benchmark_stats2 "$TRIALS"