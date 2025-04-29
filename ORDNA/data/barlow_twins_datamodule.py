import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset


class BarlowTwinsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        samples_dir: Path,
        labels_file: Path,
        sequence_length: int,
        sample_subset_size: int,
        batch_size: int = 8,
    ) -> None:
        super().__init__()
        self.samples_dir = samples_dir
        self.labels_file = labels_file
        self.sequence_length = sequence_length
        self.sample_subset_size = sample_subset_size
        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        # Legge il CSV di split con colonne 'spygen_code' e 'split'
        df = pd.read_csv(self.labels_file)
        train_codes = df.loc[df['set'] == 'train', 'spygen_code'].tolist()
        valid_codes = df.loc[df['set'] == 'validation', 'spygen_code'].tolist()

        # Costruisce i percorsi ai file .csv corrispondenti
        train_files = [self.samples_dir / f"{code}.csv" for code in train_codes]
        valid_files = [self.samples_dir / f"{code}.csv" for code in valid_codes]

        if stage == 'fit' or stage is None:
            self.train_dataset = BarlowTwinsDataset(
                sample_files=train_files,
                sample_subset_size=self.sample_subset_size,
                sequence_length=self.sequence_length,
            )
            self.val_dataset = BarlowTwinsDataset(
                sample_files=valid_files,
                sample_subset_size=self.sample_subset_size,
                sequence_length=self.sequence_length,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
