#!/bin/bash -l
#SBATCH -p smp    	 
#SBATCH --ntasks 1          
#SBATCH --cpus-per-task 50
#SBATCH --mem-per-cpu 3G  
#SBATCH --time=5-12:00:00

# Activating environment
module load anaconda/2023.07-1
conda activate testenv

# Execute simulation
srun python3 scripts_HPC_Analytical_Modelling/py_AnalyzeEgoBetweenness_paper_submitted.py

# Exit job
exit 7