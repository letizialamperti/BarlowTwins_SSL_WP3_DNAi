import bisect
import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from ORDNA.utils.sequence_mapper import SequenceMapper

class BarlowTwinsDataset(Dataset):
    def __init__(self, samples_dir: Path, labels_file: Path, sample_subset_size: int, sequence_length: int) -> None:
        self.samples_dir = samples_dir
        self.labels_dict = self.load_labels(labels_file)
        self.files = []
        self.accumulated_num_subsets = []
        self.sequence_mapper = SequenceMapper()
        self.sample_subset_size = sample_subset_size
        self.pad_seq_to_len = sequence_length

        for file in samples_dir.rglob("*.csv"):
            # Check if 'Forward' and 'Reverse' columns exist
            df = pd.read_csv(file, nrows=1)
            if 'Forward' not in df.columns or 'Reverse' not in df.columns:
                print(f"File {file} does not contain required 'Forward' and 'Reverse' columns.")
                continue
            
            self.files.append(file)
            with open(file) as f:
                sample_len = sum(1 for line in f) - 1  # Subtract 1 for header

            num_subsets = sample_len // (2 * sample_subset_size)

            if not self.accumulated_num_subsets:
                self.accumulated_num_subsets.append(num_subsets)
            else:
                self.accumulated_num_subsets.append(self.accumulated_num_subsets[-1] + num_subsets)

    def load_labels(self, file_path: Path):
        labels_df = pd.read_csv(file_path)
        return dict(zip(labels_df['spygen_code'], labels_df['protection']))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_index = bisect.bisect_left(self.accumulated_num_subsets, index + 1)
        file = self.files[sample_index]
        label = self.labels_dict[file.stem]

        if sample_index == 0:
            subset_index = index
        else:
            subset_index = index - self.accumulated_num_subsets[sample_index-1]

        loc = subset_index * self.sample_subset_size * 2
        skiprows = range(1, loc+1) # Never skip 0 (column names)
        sample_subset_df = pd.read_csv(self.files[sample_index], skiprows=skiprows, nrows=2*self.sample_subset_size)
        sample_subset1_df = sample_subset_df.iloc[:self.sample_subset_size]
        sample_subset2_df = sample_subset_df.iloc[self.sample_subset_size:]

        sample_subset1 = self._get_tensor_from_df(sample_subset1_df)
        sample_subset2 = self._get_tensor_from_df(sample_subset2_df)
        
        return sample_subset1, sample_subset2, torch.tensor(label, dtype=torch.long)

    def _pad_dataframe(self, df: pd.DataFrame, nrows: int) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame()  # Return an empty dataframe
        repeats = (nrows + len(df) - 1) // len(df)
        repeats = max(1, repeats)  # Ensure repeats is at least 1 to avoid ZeroDivisionError
        return pd.concat([df] * repeats).iloc[:nrows]

    def _get_tensor_from_df(self, df: pd.DataFrame) -> torch.Tensor:
        forward, reverse = self._get_sequences_from_df(df)
        forward = self.sequence_mapper.map_seq_list(forward, pad_to_len=self.pad_seq_to_len)
        reverse = self.sequence_mapper.map_seq_list(reverse, pad_to_len=self.pad_seq_to_len)
        return torch.stack((torch.tensor(forward), torch.tensor(reverse)), dim=1)

    def _get_sequences_from_df(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        forward_sequences = df['Forward'].to_list()
        reverse_sequences = df['Reverse'].to_list()
        
        return forward_sequences, reverse_sequences

    def __len__(self) -> int:
        return self.accumulated_num_subsets[-1]
