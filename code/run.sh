#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.

# Run the example script first
echo "Running example: py_example_betweenness_centrality_calculation.py"
python py_example_betweenness_centrality_calculation.py

# Check if SLURM is available by checking if 'sbatch' command exists
if command -v sbatch &> /dev/null
then
    echo "SLURM detected. Submitting jobs to SLURM..."
    sbatch py_AnalyzeEgoBetweenness.sh
    sbatch py_AnalyzeCentralityBasedAdvance_paper.sh
else
    echo "SLURM not detected. Running Python scripts directly..."
    python py_AnalyzeEgoBetweenness_paper_submitted.py
    python py_AnalyzeCentralityBasedAdvance_paper_submitted.py
fi
