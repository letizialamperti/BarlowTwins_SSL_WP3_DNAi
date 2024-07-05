import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        # Ensure logits and labels are on the same device
        logits = logits.to(labels.device)

        # Debugging: Print shapes and values
        print(f"Inside OrdinalCrossEntropyLoss - logits shape: {logits.shape}, labels shape: {labels.shape}")
        print(f"Inside OrdinalCrossEntropyLoss - logits: {logits}")
        print(f"Inside OrdinalCrossEntropyLoss - labels: {labels}")

        # Ensure logits and labels are within valid range
        if not torch.all(labels >= 0) or not torch.all(labels < self.num_classes):
            raise ValueError("Labels out of range")
        
        # Check for NaNs or Infs in logits
        if torch.isnan(logits).any():
            raise ValueError("Logits contain NaNs")
        if torch.isinf(logits).any():
            raise ValueError("Logits contain Infs")

        # Adjust logits for ordinal loss
        logits = logits.view(-1, self.num_classes)
        labels = labels.view(-1)

        # Debugging: Print adjusted logits and labels
        print(f"Adjusted logits shape: {logits.shape}")
        print(f"Adjusted labels shape: {labels.shape}")

        # Compute cumulative probabilities
        cum_probs = torch.sigmoid(logits)
        print(f"cum_probs shape: {cum_probs.shape}")
        cum_probs = torch.cat([cum_probs, torch.ones_like(cum_probs[:, :1])], dim=1)
        prob = cum_probs[:, :-1] - cum_probs[:, 1:]
        print(f"prob shape: {prob.shape}")

        # Ensure no values in `labels` are out of bounds
        if torch.any(labels >= prob.size(1)):
            raise ValueError("Labels out of bounds for the number of logits provided")

        # Compute one-hot labels
        one_hot_labels = torch.zeros_like(prob).scatter(1, labels.unsqueeze(1), 1)
        print(f"one_hot_labels shape: {one_hot_labels.shape}")

        # Compute loss
        loss = - (one_hot_labels * torch.log(prob + 1e-9) + (1 - one_hot_labels) * torch.log(1 - prob + 1e-9)).sum(dim=1).mean()

        return loss

class Classifier(pl.LightningModule):
    def __init__(self, barlow_twins_model: SelfAttentionBarlowTwinsEmbedder, sample_repr_dim: int, num_classes: int, initial_learning_rate: float = 1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['barlow_twins_model'])  # Save hyperparameters, but ignore barlow_twins_model
        self.barlow_twins_model = barlow_twins_model.eval()  # Set to evaluation mode
        self.num_classes = num_classes

        # Freeze the parameters of Barlow Twins model
        for param in self.barlow_twins_model.parameters():
            param.requires_grad = False

        # Classifier adjusted for dynamic number of classes
        self.classifier = nn.Sequential(
            nn.Linear(sample_repr_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes - 1)  # Output num_classes - 1
        )
        
        # Loss function
        self.loss_fn = OrdinalCrossEntropyLoss(num_classes)

        # Metrics
        if num_classes > 2:
            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.train_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
            self.val_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        else:
            self.train_accuracy = Accuracy(task="binary")
            self.val_accuracy = Accuracy(task="binary")
            self.train_conf_matrix = ConfusionMatrix(task="binary")
            self.val_conf_matrix = ConfusionMatrix(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_repr = self.barlow_twins_model.repr_module(x)  # Extract representation using Barlow Twins
        print(f"sample_repr shape: {sample_repr.shape}")  # Debugging: print the shape
        return self.classifier(sample_repr).squeeze(dim=1)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        sample_subset1, sample_subset2, labels = batch

        output1 = self(sample_subset1)
        output2 = self(sample_subset2)

        # Classification loss
        labels = labels.to(self.device)
        try:
            class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        except Exception as e:
            print(f"Error in OrdinalCrossEntropyLoss forward pass: {e}")
            print(f"logits shape: {output1.shape}, labels shape: {labels.shape}")
            raise e
        self.log('class_loss', class_loss)

        # Accuracy calculation
        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.train_accuracy(combined_preds, combined_labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)

        return class_loss

    def validation_step(self, batch, batch_idx: int):
        sample_subset1, sample_subset2, labels = batch

        output1 = self(sample_subset1)
        output2 = self(sample_subset2)

        # Classification loss
        labels = labels.to(self.device)
        try:
            class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        except Exception as e:
            print(f"Error in OrdinalCrossEntropyLoss forward pass: {e}")
            print(f"logits shape: {output1.shape}, labels shape: {labels.shape}")
            raise e

        # Combining predictions and labels
        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.val_accuracy(combined_preds, combined_labels)

        # Log the combined accuracy
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', class_loss)

        return class_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate)
        return optimizer
