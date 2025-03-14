#!/bin/bash
set -e  # Ferma lo script se si verifica un errore

# ----------------------------------------
# DIRETTIVE SCHEDULING (modifica se usi OAR, SLURM, ecc.)
# Esempio con direttive OAR (commenta o modifica se non necessario)
#OAR -n embedding-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=02:00:00
#OAR -p gpumodel='A100'
#OAR --stdout embedding-logfile.out
#OAR --stderr embedding-errorfile.err
#OAR --project pr-qiepb

# ----------------------------------------
# Attiva l'ambiente Conda (modifica il path se necessario)
source /applis/environments/conda.sh
conda activate zioboia

# Esegui lo script di visualizzazione degli embedding
echo "Starting visualize_embeddings.py..."
python visualize_embeddings.py
echo "Visualize embeddings completed."
