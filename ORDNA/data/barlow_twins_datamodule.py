import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from typing import Optional
from pathlib import Path
from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset
import numpy as np

class BarlowTwinsDataModule(pl.LightningDataModule):
    def __init__(self, samples_dir: Path, labels_file: Path, sequence_length: int, sample_subset_size: int, batch_size: int = 8, val_split: float = 0.2) -> None:
        super().__init__()

        self.samples_dir = samples_dir
        self.labels_file = labels_file
        self.sequence_length = sequence_length
        self.sample_subset_size = sample_subset_size
        self.batch_size = batch_size
        self.val_split = val_split

        assert(batch_size is not None) 

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            full_dataset = BarlowTwinsDataset(
                samples_dir=self.samples_dir,
                labels_file=self.labels_file,
                sample_subset_size=self.sample_subset_size,
                sequence_length=self.sequence_length
            )
            
            # Calcola le dimensioni del training e validation set
            dataset_size = len(full_dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.val_split * dataset_size))

            # Mescola gli indici
            np.random.shuffle(indices)

            train_indices, val_indices = indices[split:], indices[:split]

            # Crea i subset per training e validation
            self.train_dataset = Subset(full_dataset, train_indices)
            self.val_dataset = Subset(full_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=12, 
            pin_memory=torch.cuda.is_available(), 
            drop_last=False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=12, 
            pin_memory=torch.cuda.is_available(), 
            drop_last=False
        )
