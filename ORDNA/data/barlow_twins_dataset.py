import bisect
import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from ORDNA.utils.sequence_mapper import SequenceMapper


class BarlowTwinsDataset(Dataset):
    def __init__(
        self,
        sample_files: List[Path],
        sample_subset_size: int,
        sequence_length: int,
    ) -> None:
        # Lista di file già filtrata dal DataModule
        self.files: List[Path] = []
        self.accumulated_num_subsets: List[int] = []
        self.sample_subset_size = sample_subset_size
        self.pad_seq_to_len = sequence_length
        self.sequence_mapper = SequenceMapper()

        running = 0
        for file in sample_files:
            # Verifica colonne minime
            df0 = pd.read_csv(file, nrows=1)
            if not {'Forward', 'Reverse'}.issubset(df0.columns):
                print(f"Skip {file.name}: manca Forward/Reverse")
                continue

            # Conta righe dati (escluse header)
            with open(file) as f:
                n = sum(1 for _ in f) - 1

            num_chunks = n // (2 * sample_subset_size)
            if num_chunks <= 0:
                continue

            self.files.append(file)
            running += num_chunks
            self.accumulated_num_subsets.append(running)

    def __len__(self) -> int:
        return self.accumulated_num_subsets[-1] if self.accumulated_num_subsets else 0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trova file e chunk corrispondente
        file_idx = bisect.bisect_left(self.accumulated_num_subsets, index + 1)
        file = self.files[file_idx]
        prev_cum = self.accumulated_num_subsets[file_idx-1] if file_idx > 0 else 0
        subset_idx = index - prev_cum

        # Offset righe da saltare (1 header + dati precedenti)
        loc = subset_idx * self.sample_subset_size * 2
        skip = range(1, loc + 1)

        # Leggi 2*sample_subset_size righe
        df_chunk = pd.read_csv(file, skiprows=skip, nrows=2 * self.sample_subset_size)
        df1 = df_chunk.iloc[:self.sample_subset_size]
        df2 = df_chunk.iloc[self.sample_subset_size:]

        return self._to_tensor(df1), self._to_tensor(df2)

    def _to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        # Filtra righe senza sequenza
        df = df[df['Forward'].astype(str).str.strip().ne("") &
                df['Reverse'].astype(str).str.strip().ne("")]
        forward = df['Forward'].tolist()
        reverse = df['Reverse'].tolist()

        # Mappatura + padding
        fwd_ids = self.sequence_mapper.map_seq_list(forward, pad_to_len=self.pad_seq_to_len)
        rev_ids = self.sequence_mapper.map_seq_list(reverse, pad_to_len=self.pad_seq_to_len)

        # Torna tensor di forma (sequence_length, 2)
        return torch.stack((torch.tensor(fwd_ids), torch.tensor(rev_ids)), dim=1)
