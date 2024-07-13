from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset
from pathlib import Path

# Configura i percorsi e i parametri
samples_dir = Path("/store/sdsc/sd29/letizia/sud_corse/train")
labels_file = Path("label/ordinal_label_Sud_Corse.csv")
sequence_length = 300
sample_subset_size = 500

# Crea il dataset
train_dataset = BarlowTwinsDataset(
    samples_dir=samples_dir,
    labels_file=labels_file,
    sample_subset_size=sample_subset_size,
    sequence_length=sequence_length
)

# Itera sul dataset per verificare che tutto funzioni correttamente
for idx, (sample1, sample2, label) in enumerate(train_dataset):
    print(f"Sample {idx}: Label {label}")
