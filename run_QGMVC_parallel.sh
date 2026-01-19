#!/bin/bash
#SBATCH --job-name=QGMVC
#SBATCH --time=30-00:00:00        # max 30 days
#SBATCH --output=%j.out            # stdout to JOBID.out
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 1 Python process per job
#SBATCH --cpus-per-task=30      # number of cores to use (adjust!)
#SBATCH --mem=64G                  # memory per node (adjust as needed)

# =========================
# Load environment
# =========================
# Activate your Python environment
source /path/to/quantumreservoirpy/bin/activate   # <-- change path

# =========================
# Prevent thread oversubscription
# =========================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# =========================
# Run the Python script
# =========================
# Pass all command-line arguments ($@)
python QGMVC_parallel.py "$@"
