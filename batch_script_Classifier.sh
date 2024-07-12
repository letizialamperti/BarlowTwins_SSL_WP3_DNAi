#!/bin/bash -l

# SLURM SBATCH Directives
#SBATCH --job-name=classifier-job
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --output=classifier-logfile-%j.log
#SBATCH --error=classifier-errorfile-%j.err
#SBATCH --account=sd29

# Module loading section
module load daint-gpu
module load cray-python

# Activate conda environment
source activate diaus_1

# Define dataset and labels path
DATASET_DIR="/store/sdsc/sd29/letizia/dataset_5_levels_460"
LABELS_FILE="label/labels_5_levels.csv"

# Command to run the Python script
echo "Starting the training process."
export CUDA_LAUNCH_BLOCKING=1
srun -ul $HOME/miniconda3/envs/diaus_1/bin/python training_Classifier.py \
    --arg_log True \
    --samples_dir $DATASET_DIR \
    --labels_file $LABELS_FILE \
    --embedder_type barlow_twins \
    --sequence_length 300 \
    --sample_subset_size 500 \
    --num_classes 5 \
    --batch_size 32 \
    --token_emb_dim 8 \
    --sample_repr_dim 512 \
    --sample_emb_dim 128 \
    --initial_learning_rate 1e-3 \
    --max_epochs 2

echo "Training completed successfully."
