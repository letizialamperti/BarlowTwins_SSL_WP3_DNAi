import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder  # Import the Barlow Twins model

class WeightedOrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(WeightedOrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        logits = logits.view(-1, self.num_classes - 1).to(labels.device)
        labels = labels.view(-1, 1).to(logits.device)

        # Ensure labels are within valid range
        assert labels.min() >= 0 and labels.max() < self.num_classes, f"Labels are out of valid range: {labels}"

        cum_probs = torch.sigmoid(logits)
        cum_probs = torch.cat([cum_probs, torch.ones_like(cum_probs[:, :1])], dim=1)
        prob = cum_probs[:, :-1] - cum_probs[:, 1:]

        class_counts = torch.bincount(labels.view(-1), minlength=self.num_classes).float().to(logits.device)
        weights = class_counts / class_counts.sum()

        # Handle cases where some classes may not be present in the batch
        weights = torch.where(weights == 0, torch.tensor(1.0, device=weights.device), weights)
        inv_weights = 1.0 / weights
        inv_weights = inv_weights / inv_weights.sum()

        # Compute the loss
        loss = - (inv_weights[labels.view(-1)] * torch.log(prob.gather(1, labels) + 1e-9)).mean()

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
            nn.Linear(1024, 1 if num_classes == 2 else num_classes - 1)  # Output 1 if binary classification
        )
        
        # Loss function
        self.loss_fn = WeightedOrdinalCrossEntropyLoss(num_classes=num_classes)

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
        logits = self.classifier(sample_repr).squeeze(dim=1)
        print(f"logits shape: {logits.shape}")  # Debugging: print the shape
        return logits

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        sample_subset1, sample_subset2, labels = batch

        output1 = self(sample_subset1)
        output2 = self(sample_subset2)

        print(f"labels shape before loss: {labels.shape}")
        print(f"logits shape: {output1.shape}")

        # Classification loss
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        self.log('class_loss', class_loss)

        # Accuracy calculation
        if self.num_classes > 2:
            pred1 = torch.argmax(output1, dim=1)
            pred2 = torch.argmax(output2, dim=1)
        else:
            pred1 = (torch.sigmoid(output1) > 0.5).long()
            pred2 = (torch.sigmoid(output2) > 0.5).long()

        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.train_accuracy(combined_preds, combined_labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)

        return class_loss

    def validation_step(self, batch, batch_idx: int):
        sample_subset1, sample_subset2, labels = batch

        output1 = self(sample_subset1)
        output2 = self(sample_subset2)

        print(f"validation labels shape before loss: {labels.shape}")
        print(f"logits shape: {output1.shape}")

        # Classification loss
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)

        # Combining predictions and labels
        if self.num_classes > 2:
            pred1 = torch.argmax(output1, dim=1)
            pred2 = torch.argmax(output2, dim=1)
        else:
            pred1 = (torch.sigmoid(output1) > 0.5).long()
            pred2 = (torch.sigmoid(output2) > 0.5).long()

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
