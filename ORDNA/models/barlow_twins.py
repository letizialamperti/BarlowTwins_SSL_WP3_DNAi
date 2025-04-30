import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from typing import Tuple
from ORDNA.models.representation_module import SelfAttentionRepresentationModule

class SelfAttentionBarlowTwinsEmbedder(pl.LightningModule):
    def __init__(
        self,
        token_emb_dim: int,
        seq_len: int,
        sample_repr_dim: int,
        sample_emb_dim: int,
        lmbda: float = 0.005,
        initial_learning_rate: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        # Representation module
        self.repr_module = SelfAttentionRepresentationModule(
            token_emb_dim=token_emb_dim,
            seq_len=seq_len,
            repr_dim=sample_repr_dim
        )
        # MLP for creating sample embeddings
        self.out_mlp = nn.Sequential(
            nn.Linear(sample_repr_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, sample_emb_dim)
        )
        self.lmbda = lmbda
        self.initial_learning_rate = initial_learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, sequence_length, ...)
        sample_repr = self.repr_module(x)          # → (B, sample_repr_dim)
        sample_emb = self.out_mlp(sample_repr)     # → (B, sample_emb_dim)
        return sample_emb

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x1, x2 = batch
        B = x1.size(0)

        z1 = self(x1)
        z2 = self(x2)

        # normalize
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        # cross-correlation
        c = (z1.T @ z2) / B
        diag_loss     = ((torch.diag(c) - 1) ** 2).sum()
        off_diag_loss = ((c - torch.diag(torch.diag(c))) ** 2).sum() * self.lmbda
        loss = diag_loss + off_diag_loss

        self.log('train_barlow_loss',     loss,     prog_bar=True)
        self.log('train_diag_loss',       diag_loss)
        self.log('train_off_diag_loss',   off_diag_loss)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x1, x2 = batch
        B = x1.size(0)

        z1 = self(x1)
        z2 = self(x2)

        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        c = (z1.T @ z2) / B
        diag_loss     = ((torch.diag(c) - 1) ** 2).sum()
        off_diag_loss = ((c - torch.diag(torch.diag(c))) ** 2).sum() * self.lmbda
        loss = diag_loss + off_diag_loss

        self.log('val_barlow_loss',     loss,     prog_bar=True)
        self.log('val_diag_loss',       diag_loss)
        self.log('val_off_diag_loss',   off_diag_loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.initial_learning_rate)
