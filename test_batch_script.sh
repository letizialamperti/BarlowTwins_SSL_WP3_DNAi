#!/bin/bash -l

# SLURM SBATCH Directives
#SBATCH --job-name=letizias-job
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --output=letizias-logfile-%j.log
#SBATCH --error=letizias-errorfile-%j.err
#SBATCH --account=sd29

# Module loading section
module load daint-gpu
module load cray-python




# Command to run the Python script
echo "Starting the training process."
srun -ul $HOME/miniconda3/envs/diaus_1/bin/python test_classifier.py  

echo "Visualizing script completed successfully."
