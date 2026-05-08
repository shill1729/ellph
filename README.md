# Ellipsoidal Intersection Benchmarking

This repository contains an experiment for benchmarking five algorithms that compute the optimal intersection radius of non-homogeneous ellipsoids in arbitrary dimension. The C++ implementation generates random ellipsoid instances, evaluates each solver across a grid of values for the number of ellipsoids and the ambient dimension, measures runtimes, and writes empirical statistics to a CSV file. A separate Python script makes plots and tables out of it.

## Dependencies and Versions

The C++ code has been tested on macOS with the following packages installed via Homebrew:

- NLopt 2.10.0
- Boost 1.89.0_1
- Eigen 5.0.1
- ALGLIB C++ 4.07.0

Install these with homebrew via

    brew install nlopt boost eigen

Download ALGLIB C++ from the ALGLIB website and place the extracted source tree at:

    third_party/alglib-cpp

The repository ignores this directory, so it is a local dependency rather than redistributed source. CMake compiles the ALGLIB sources needed by the SOCP solver. You can also keep ALGLIB elsewhere and pass its path explicitly:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DALGLIB_ROOT=/path/to/alglib-cpp

The Python script uses the following versions:

- Python 3.13
- NumPy 2.3.4
- pandas 2.3.3
- jinja2 3.1.6
- Matplotlib 3.10.7


## Building and running with CMake

From the repository root, configure and build:

    mkdir -p build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
    ./build/output/benchmark_stats2

This produces and runs the executable and you can pass different a number of trials:

    build/output/benchmark_stats2 50
    build/output/benchmark_stats2 100

## Create plots and tables
Now create a virtual environment
    
    # Create a virtual environment named 'venv' (or whatever name you prefer)
    python3.13 -m venv venv
    # Activate it
    source venv/bin/activate
    # Install the specific versions
    pip install numpy==2.3.4 pandas==2.3.3 matplotlib==3.10.7 jinja2==3.1.6


Now you can simply run

    python make_plots.py

To automatically get plots saved into figs/ and tables/.


# Windows
Install vcpkg

    git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
    cd C:\vcpkg
    .\bootstrap-vcpkg.bat
    .\vcpkg integrate install
    setx PATH "$($env:PATH);C:\vcpkg"

Close and reopen your terminal.
Then install eigen, boost and nlopt

    vcpkg install eigen3:x64-windows boost:x64-windows nlopt:x64-windows

Then just replace
    
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

with

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

and

    ./build/output/benchmark_stats2

with

    .\build\output\Release\benchmark_stats2.exe

For the python script do this to start instead:

    python -m venv venv
    venv\Scripts\activate
