#!/bin/bash
set -e
mkdir -p build
g++-14 -O3 -fopenmp *.cpp -o ns2d
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
./ns2d

python3 plot_density.py
python3 analysis_spectrum.py
python3 analysis_divergence.py
python3 analysis_conservation.py
python3 analysis_summary.py
python3 animate_amr.py
