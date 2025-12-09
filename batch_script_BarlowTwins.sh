#!/bin/bash
set -e  # Ferma il job in caso di errore

#OAR -n barlow-twins-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout barlow-twins-logfile-%j.out
#OAR --stderr barlow-twins-errorfile-%j.err
#OAR --project pr-qiepb


# Attivare Conda
source /applis/environments/conda.sh
conda activate zioboia

# Definire percorsi dataset e labels
DATASET_DIR="/bettik/PROJECTS/pr-qiepb/lampertl/all"
LABELS_FILE="label/split_corse_5_fold_01.csv"

# Eseguire lo script Python
echo "Starting the training process."
python training_BarlowTwins.py \
    --arg_log True \
    --samples_dir $DATASET_DIR \
    --labels_file $LABELS_FILE \
    --embedder_type barlow_twins \
    --sequence_length 300 \
    --sample_subset_size 500 \
    --num_classes 5 \
    --batch_size 8 \
    --token_emb_dim 8 \
    --sample_repr_dim 268324 \
    --sample_emb_dim 134162 \
    --barlow_twins_lambda 0.0005 \
    --initial_learning_rate 1e-3 \
    --max_epochs 1

echo "Training completed successfully."
