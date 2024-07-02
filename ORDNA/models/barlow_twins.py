import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from typing import List, Tuple
from ORDNA.models.representation_module import SelfAttentionRepresentationModule

class SelfAttentionBarlowTwinsEmbedder(pl.LightningModule):
    def __init__(self, token_emb_dim: int, seq_len: int, sample_repr_dim: int, sample_emb_dim: int,
                 lmbda: float = 0.005, initial_learning_rate: float = 1e-5):
        super().__init__()
        
        # Representation module
        self.repr_module = SelfAttentionRepresentationModule(token_emb_dim=token_emb_dim,
                                                             seq_len=seq_len,
                                                             repr_dim=sample_repr_dim)
        
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
        
        # Other parameters
        self.lmbda = lmbda
        self.initial_learning_rate = initial_learning_rate

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_repr = self.repr_module(x)  # B x sample_repr_dim
        sample_emb = self.out_mlp(sample_repr)  # B x sample_emb_dim
        return sample_emb

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        sample_subset1, sample_subset2, labels = batch
        batch_size = sample_subset1.size(0)

        subset1_emb = self(sample_subset1)
        subset2_emb = self(sample_subset2)

        # Barlow Twins loss
        subset1_emb_norm = (subset1_emb - torch.mean(subset1_emb, dim=0)) / torch.std(subset1_emb, dim=0)
        subset2_emb_norm = (subset2_emb - torch.mean(subset2_emb, dim=0)) / torch.std(subset2_emb, dim=0)

        cross_corr_mat = torch.matmul(torch.transpose(subset1_emb_norm, 0, 1), subset2_emb_norm) / batch_size
        diag = torch.diag(cross_corr_mat)
        diag_loss = torch.sum(torch.square(diag - torch.ones_like(diag)))

        off_diag = cross_corr_mat - torch.diag(diag)
        off_diag_loss = torch.sum(torch.square(off_diag)) * self.lmbda

        barlow_loss = diag_loss + off_diag_loss
        self.log('train_barlow_loss', barlow_loss)
        self.log('train_diag_loss', diag_loss)
        self.log('train_off_diag_loss', off_diag_loss)

        return barlow_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        sample_subset1, sample_subset2, labels = batch
        batch_size = sample_subset1.size(0)

        subset1_emb = self(sample_subset1)
        subset2_emb = self(sample_subset2)

        # Barlow Twins loss
        subset1_emb_norm = (subset1_emb - torch.mean(subset1_emb, dim=0)) / torch.std(subset1_emb, dim=0)
        subset2_emb_norm = (subset2_emb - torch.mean(subset2_emb, dim=0)) / torch.std(subset2_emb, dim=0)

        cross_corr_mat = torch.matmul(torch.transpose(subset1_emb_norm, 0, 1), subset2_emb_norm) / batch_size
        diag = torch.diag(cross_corr_mat)
        diag_loss = torch.sum(torch.square(diag - torch.ones_like(diag)))

        off_diag = cross_corr_mat - torch.diag(diag)
        off_diag_loss = torch.sum(torch.square(off_diag)) * self.lmbda

        barlow_loss = diag_loss + off_diag_loss
        self.log('val_barlow_loss', barlow_loss)
        self.log('val_diag_loss', diag_loss)
        self.log('val_off_diag_loss', off_diag_loss)

        return barlow_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.initial_learning_rate)
        return optimizer
