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
        logits = logits.view(-1, self.num_classes - 1)
        labels = labels.view(-1, 1).to(logits.device)  # Ensure labels are on the same device as logits

        # Compute the cumulative probabilities
        cum_probs = torch.sigmoid(logits)
        cum_probs = torch.cat([cum_probs, torch.ones_like(cum_probs[:, :1])], dim=1)
        prob = cum_probs[:, :-1] - cum_probs[:, 1:]

        # Compute weights based on the presence of labels in the batch
        class_counts = torch.bincount(labels.view(-1), minlength=self.num_classes).float()
        class_counts[class_counts == 0] = 1  # Avoid division by zero
        weights = 1.0 / class_counts
        weights = weights[labels.view(-1)]

        one_hot_labels = torch.zeros_like(prob).scatter(1, labels, 1)
        loss = - (one_hot_labels * torch.log(prob + 1e-9) + (1 - one_hot_labels) * torch.log(1 - prob + 1e-9)).sum(dim=1)

        # Apply weights to the loss
        loss = (loss * weights).mean()

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
            nn.Linear(1024, num_classes - 1)  # Output num_classes - 1 for ordinal classification
        )
        
        # Loss function
        if num_classes > 2:
            self.loss_fn = WeightedOrdinalCrossEntropyLoss(num_classes=num_classes)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

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
        return self.classifier(sample_repr).squeeze(dim=1)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        sample_subset1, sample_subset2, labels = batch

        output1 = self(sample_subset1)
        output2 = self(sample_subset2)
        
        # Classification loss
        if self.num_classes > 2:
            class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        else:
            labels = labels.float()
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

        # Classification loss
        if self.num_classes > 2:
            class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
            pred1 = torch.argmax(output1, dim=1)
            pred2 = torch.argmax(output2, dim=1)
        else:
            labels = labels.float()
            class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
            pred1 = (torch.sigmoid(output1) > 0.5).long()
            pred2 = (torch.sigmoid(output2) > 0.5).long()

        # Combining predictions and labels
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
