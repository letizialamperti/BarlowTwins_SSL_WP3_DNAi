from torch.utils.data import DataLoader
import torch

class BarlowTwinsDataModule(pl.LightningDataModule):
    def __init__(self, samples_dir: Path, labels_file: Path, sequence_length: int, sample_subset_size: int, batch_size: int = 8) -> None:
        super().__init__()
        self.train_samples_dir = samples_dir / "train"
        self.val_samples_dir = samples_dir / "valid"
        self.labels_file = labels_file
        self.sequence_length = sequence_length
        self.sample_subset_size = sample_subset_size
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = BarlowTwinsDataset(
                samples_dir=self.train_samples_dir,
                labels_file=self.labels_file,
                sample_subset_size=self.sample_subset_size,
                sequence_length=self.sequence_length
            )

            self.val_dataset = BarlowTwinsDataset(
                samples_dir=self.val_samples_dir,
                labels_file=self.labels_file,
                sample_subset_size=self.sample_subset_size,
                sequence_length=self.sequence_length
            )

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

    def calculate_class_weights(self, num_classes):
        print("Calculating class weights...")
        labels = []
        for _, _, label in self.train_dataset:
            labels.append(label)
        labels = torch.tensor(labels)
        class_counts = torch.bincount(labels, minlength=num_classes)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
        print(f"Class weights: {class_weights}")
        return class_weights
