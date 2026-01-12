#!/bin/bash
#SBATCH --job-name=QGMVC
#SBATCH --time=30-00:00:00        # 30 days max
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1

# Activate your Python environment
source quantumreservoirpy/bin/activate   # adjust path if needed

# Run the script with all command-line arguments passed
python QGMVC.py "$@"
