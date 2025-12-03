#!/bin/bash
#SBATCH --job-name=compile_engine
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --output=compile_engine.%j


module restore deps

source venv/bin/activate

c++ -std=c++20 -O3 -shared -fPIC $(python -m pybind11 --includes) infini_gram/cpp_engine.cpp -o infini_gram/cpp_engine$(python-config --extension-suffix)
