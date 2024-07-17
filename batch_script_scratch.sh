#!/bin/bash -l

# SLURM SBATCH Directives
#SBATCH --job-name=classifier-job
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --output=scratch_classifier-logfile-%j.log
#SBATCH --error=scratch_classifier-errorfile-%j.err
#SBATCH --account=sd29

# Module loading section
module load daint-gpu
module load cray-python

# Activate conda environment
source activate diaus_1

# Define dataset and labels path
DATASET_DIR="/store/sdsc/sd29/letizia/sud_corse"
LABELS_FILE="label/ordinal_label_Sud_Corse.csv"

# Command to run the Python script
echo "Starting the training process."
export CUDA_LAUNCH_BLOCKING=1
srun -ul $HOME/miniconda3/envs/diaus_1/bin/python training_classifier_from_scratch.py \
    --arg_log True \
    --samples_dir $DATASET_DIR \
    --labels_file $LABELS_FILE \
    --sequence_length 300 \
    --sample_subset_size 500 \
    --num_classes 4 \
    --batch_size 32 \
    --token_emb_dim 8 \
    --sample_repr_dim 128 \
    --sample_emb_dim 64 \
    --initial_learning_rate 1e-3 \
    --max_epochs 2

echo "Training completed successfully."
