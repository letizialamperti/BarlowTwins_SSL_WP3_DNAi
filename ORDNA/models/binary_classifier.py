import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
import wandb  # Importa Wandb per il logging

def calculate_class_weights(dataset, num_classes):
    labels = []
    for _, _, label in dataset:
        labels.append(label)
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    return class_weights

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, logits, labels):
        logits = logits.to(labels.device)
        labels = labels.view(-1).float()
        if self.class_weights is not None:
            class_weights = self.class_weights[labels.long()].to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=class_weights)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

class BinaryClassifier(pl.LightningModule):
    def __init__(self, barlow_twins_model: SelfAttentionBarlowTwinsEmbedder, sample_repr_dim: int, initial_learning_rate: float = 1e-5, train_dataset=None):
        super().__init__()
        self.save_hyperparameters(ignore=['barlow_twins_model', 'train_dataset'])
        self.barlow_twins_model = barlow_twins_model.eval()
        self.num_classes = 2  # For binary classification
        for param in self.barlow_twins_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(sample_repr_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1)  # Output a single value for binary classification
        )
        self.class_weights = calculate_class_weights(train_dataset, self.num_classes).to(self.device) if train_dataset is not None else None
        self.loss_fn = BinaryCrossEntropyLoss(self.class_weights)
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_conf_matrix = ConfusionMatrix(task="binary")
        self.val_conf_matrix = ConfusionMatrix(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_repr = self.barlow_twins_model.repr_module(x)
        return self.classifier(sample_repr).squeeze(dim=1)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        sample_subset1, sample_subset2, labels = batch
        if torch.any(labels >= self.num_classes):
            raise ValueError("Labels out of range")
        output1 = self(sample_subset1)
        output2 = self(sample_subset2)
        labels = labels.to(self.device)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        self.log('class_loss', class_loss)
        pred1 = torch.sigmoid(output1) > 0.5
        pred2 = torch.sigmoid(output2) > 0.5
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.train_accuracy(combined_preds, combined_labels)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        precision = self.train_precision(combined_preds, combined_labels)
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        recall = self.train_recall(combined_preds, combined_labels)
        self.log('train_recall', recall, on_step=True, on_epoch=True)
        return class_loss

    def validation_step(self, batch, batch_idx: int):
        sample_subset1, sample_subset2, labels = batch
        if torch.any(labels >= self.num_classes):
            raise ValueError("Labels out of range")
        output1 = self(sample_subset1)
        output2 = self(sample_subset2)
        labels = labels.to(self.device)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        pred1 = torch.sigmoid(output1) > 0.5
        pred2 = torch.sigmoid(output2) > 0.5
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.val_accuracy(combined_preds, combined_labels)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True)
        self.log('val_loss', class_loss)
        precision = self.val_precision(combined_preds, combined_labels)
        self.log('val_precision', precision, on_step=True, on_epoch=True)
        recall = self.val_recall(combined_preds, combined_labels)
        self.log('val_recall', recall, on_step=True, on_epoch=True)
        return class_loss

    def log_conf_matrix(self, conf_matrix, stage):
        conf_matrix = conf_matrix.cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(self.num_classes), yticklabels=range(self.num_classes))
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'{stage.capitalize()} Confusion Matrix')
        plt.close(fig)
        self.logger.experiment.log({f"{stage}_conf_matrix": wandb.Image(fig)})

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.initial_learning_rate, total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]
