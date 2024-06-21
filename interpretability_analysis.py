import torch
import numpy as np
import pandas as pd
from pathlib import Path
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.sequence_mapper import SequenceMapper

# Checkpoint and dataset paths
CHECKPOINT_PATH = Path('lightning_logs/ORDNA/yakukynx/checkpoints/epoch=9-step=1120.ckpt')
DATASET = 'Sud_Corse'
SAMPLE_DIR = Path(f'/store/sdsc/sd29/letizia/dataset_5_levels_460')
SEQUENCE_LENGTH = 300
SAMPLE_SUBSET_SIZE = 1000
NUM_CLASSES = 2  # Adjust this based on the number of classes in your classifier

# Load the model
model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(CHECKPOINT_PATH, num_classes=NUM_CLASSES)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize SequenceMapper
sequence_mapper = SequenceMapper()

# Integrated Gradients
ig = IntegratedGradients(model)

def prepare_input(sample_df, sequence_length, sample_subset_size):
    sample_df = sample_df.sample(frac=1)  # Random shuffle
    input_tensor_list = []

    for start in range(0, len(sample_df), sample_subset_size):
        end = min(start + sample_subset_size, len(sample_df))
        batch_df = sample_df.iloc[start:end]
        forward_sequences = batch_df['Forward'].tolist()
        reverse_sequences = batch_df['Reverse'].tolist()

        # Convert sequence data to tensor
        forward_tensor = torch.tensor(sequence_mapper.map_seq_list(forward_sequences, sequence_length))
        reverse_tensor = torch.tensor(sequence_mapper.map_seq_list(reverse_sequences, sequence_length))
        input_tensor = torch.stack((forward_tensor, reverse_tensor), dim=1).to(device)
        input_tensor = input_tensor.unsqueeze(0)  # Adjusting the batch dimension

        input_tensor_list.append(input_tensor)

    return input_tensor_list

# Example function to apply Integrated Gradients and get attributions
def get_attributions(input_tensor, target_class):
    input_tensor.requires_grad = True
    attributions, delta = ig.attribute(input_tensor, target=target_class, return_convergence_delta=True)
    return attributions, delta

# Loop through all CSV files in the directory
for i, sample_path in enumerate(SAMPLE_DIR.glob("*.csv")):
    sample_df = pd.read_csv(sample_path)
    input_tensors = prepare_input(sample_df, SEQUENCE_LENGTH, SAMPLE_SUBSET_SIZE)

    # Get attributions for each subset
    target_class = 0  # Choose the class for which you want to compute attributions
    all_attributions = []
    for input_tensor in input_tensors:
        attributions, delta = get_attributions(input_tensor, target_class)
        all_attributions.append(attributions.cpu().detach().numpy())

    # Aggregate attributions if needed
    mean_attributions = np.mean(all_attributions, axis=0)

    # Visualize attributions for the first input tensor
    attr = mean_attributions[0]
    viz.visualize_image_attr(attr, method='heat_map', show_colorbar=True, title=f'Integrated Gradients - Sample {i+1}')

    # Optionally save the attributions or results
    output_folder = "/scratch/snx3000/llampert/interpretability/"
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, f"attributions_sample_{i+1}.npy"), mean_attributions)

    print(f'Processed {i+1}/{len(list(SAMPLE_DIR.glob("*.csv")))} files')

print("Interpretability analysis completed.")
