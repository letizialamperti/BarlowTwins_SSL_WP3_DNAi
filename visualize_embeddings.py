import torch
import pandas as pd
import numpy as np
import os
import csv
from pathlib import Path
from IPython.display import display, clear_output
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.sequence_mapper import SequenceMapper

MODEL_TYPE = 'barlow_twins' 
CHECKPOINT_PATH = Path('checkpoints/BT_2025_64dim_-epoch=00-v1.ckpt')
DATASET = '460_all_data'
SAMPLE_DIR = Path('/bettik/PROJECTS/pr-qiepb/lampertl/train')
SEQUENCE_LENGTH = 300
SAMPLE_SUBSET_SIZE = 500
SAMPLE_EMB_DIM = 64  # Nuovo parametro per la dimensione dell'embedding

# Caricamento del modello; non passiamo più num_classes
if MODEL_TYPE == 'barlow_twins':
    model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(CHECKPOINT_PATH)
else:
    raise Exception('Unknown model type:', MODEL_TYPE)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Imposta cartella e file di output
version = CHECKPOINT_PATH.parents[1].name
output_folder = "BT_output/train"
os.makedirs(output_folder, exist_ok=True)
output_csv_file = os.path.join(output_folder, f"embedding_coords_{DATASET.lower()}_{version}.csv")

sequence_mapper = SequenceMapper()

# Preparazione del CSV: header dinamico basato su SAMPLE_EMB_DIM
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ["Sample"]
    header += [f"Dim{i+1}" for i in range(SAMPLE_EMB_DIM)]
    header += [f"Standard_dev_{i+1}" for i in range(SAMPLE_EMB_DIM)]
    writer.writerow(header)

    num_files = len(list(SAMPLE_DIR.rglob('*.csv')))
    for i, file_path in enumerate(SAMPLE_DIR.rglob('*.csv')):
        sample_name = file_path.stem
        sample_df = pd.read_csv(file_path)
        sample_df = sample_df.sample(frac=1)  # Mescola casualmente le righe

        sample_emb_coords = []
        # Suddividi il DataFrame in blocchi per gestire grandi dataset
        for start in range(0, len(sample_df), SAMPLE_SUBSET_SIZE):
            end = min(start + SAMPLE_SUBSET_SIZE, len(sample_df))
            batch_df = sample_df.iloc[start:end]
            forward_sequences = batch_df['Forward'].tolist()
            reverse_sequences = batch_df['Reverse'].tolist()

            # Converti le sequenze in tensori utilizzando il SequenceMapper
            forward_tensor = torch.tensor(sequence_mapper.map_seq_list(forward_sequences, SEQUENCE_LENGTH))
            reverse_tensor = torch.tensor(sequence_mapper.map_seq_list(reverse_sequences, SEQUENCE_LENGTH))
            # Impila lungo la dimensione "direzione" e aggiungi la dimensione batch
            input_tensor = torch.stack((forward_tensor, reverse_tensor), dim=1).to(device)
            input_tensor = input_tensor.unsqueeze(0)  # Forma: [1, N, 2, SEQUENCE_LENGTH]

            with torch.no_grad():
                # Ottieni l'embedding (la parte di riduzione di dimensione)
                sample_emb = model(input_tensor)
                sample_emb_coords.append(sample_emb.squeeze().cpu().numpy())

        if len(sample_emb_coords) > 0:
            sample_emb_coords = np.array(sample_emb_coords)
            mean_coords = np.mean(sample_emb_coords, axis=0)
            std_coords = np.std(sample_emb_coords, axis=0)
        else:
            mean_coords = np.zeros(SAMPLE_EMB_DIM)
            std_coords = np.zeros(SAMPLE_EMB_DIM)

        # Scrive il risultato nel CSV
        row = [sample_name] + list(mean_coords) + list(std_coords)
        writer.writerow(row)
        display(f'Processed {i+1}/{num_files} files')
        clear_output(wait=True)

print(f"File CSV creato con successo: {output_csv_file}")
