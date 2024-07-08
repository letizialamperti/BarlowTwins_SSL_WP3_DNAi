import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        # Ensure logits and labels are on the same device
        logits = logits.to(labels.device)

        # Adjust logits for ordinal loss
        logits = logits.view(-1, self.num_classes)
        labels = labels.view(-1)

        # Compute cumulative probabilities
        cum_probs = torch.sigmoid(logits)
        cum_probs = torch.cat([cum_probs, torch.ones_like(cum_probs[:, :1])], dim=1)
        prob = cum_probs[:, :-1] - cum_probs[:, 1:]

        # Compute one-hot labels
        one_hot_labels = torch.zeros_like(prob).scatter(1, labels.unsqueeze(1), 1)

        # Compute loss
        epsilon = 1e-9
        prob = torch.clamp(prob, min=epsilon, max=1-epsilon)  # Add epsilon to avoid log(0)
        loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1).mean()

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
            nn.Linear(1024, num_classes)  # Output num_classes
        )
        
        # Loss function
        self.loss_fn = OrdinalCrossEntropyLoss(num_classes)

        # Metrics
        if num_classes > 2:
            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.train_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
            self.val_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
            self.train_precision = Precision(task="multiclass", num_classes=num_classes)
            self.val_precision = Precision(task="multiclass", num_classes=num_classes)
            self.train_recall = Recall(task="multiclass", num_classes=num_classes)
            self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        else:
            self.train_accuracy = Accuracy(task="binary")
            self.val_accuracy = Accuracy(task="binary")
            self.train_conf_matrix = ConfusionMatrix(task="binary")
            self.val_conf_matrix = ConfusionMatrix(task="binary")
            self.train_precision = Precision(task="binary")
            self.val_precision = Precision(task="binary")
            self.train_recall = Recall(task="binary")
            self.val_recall = Recall(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_repr = self.barlow_twins_model.repr_module(x)  # Extract representation using Barlow Twins
        return self.classifier(sample_repr).squeeze(dim=1)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        sample_subset1, sample_subset2, labels = batch

        # Ensure labels are within the correct range
        if torch.any(labels >= self.num_classes):
            raise ValueError("Labels out of range")

        output1 = self(sample_subset1)
        output2 = self(sample_subset2)

        # Classification loss
        labels = labels.to(self.device)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        self.log('class_loss', class_loss)

        # Accuracy calculation
        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.train_accuracy(combined_preds, combined_labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        
        # Confusion Matrix
        conf_matrix = self.train_conf_matrix(combined_preds, combined_labels)
        self.log('train_conf_matrix', conf_matrix, on_step=False, on_epoch=True)
        
        # Precision
        precision = self.train_precision(combined_preds, combined_labels)
        self.log('train_precision', precision, on_step=False, on_epoch=True)
        
        # Recall
        recall = self.train_recall(combined_preds, combined_labels)
        self.log('train_recall', recall, on_step=False, on_epoch=True)

        return class_loss

    def validation_step(self, batch, batch_idx: int):
        sample_subset1, sample_subset2, labels = batch

        # Ensure labels are within the correct range
        if torch.any(labels >= self.num_classes):
            raise ValueError("Labels out of range")

        output1 = self(sample_subset1)
        output2 = self(sample_subset2)

        # Classification loss
        labels = labels.to(self.device)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)

        # Combining predictions and labels
        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.val_accuracy(combined_preds, combined_labels)
        
        # Log the combined accuracy
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', class_loss)
        
        # Confusion Matrix
        conf_matrix = self.val_conf_matrix(combined_preds, combined_labels)
        self.log('val_conf_matrix', conf_matrix, on_step=False, on_epoch=True)
        
        # Precision
        precision = self.val_precision(combined_preds, combined_labels)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        
        # Recall
        recall = self.val_recall(combined_preds, combined_labels)
        self.log('val_recall', recall, on_step=False, on_epoch=True)

        return class_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate)
        return optimizer
