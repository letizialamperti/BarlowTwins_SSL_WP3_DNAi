import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, logits, labels):
        logits = logits.to(labels.device).view(-1)
        labels = labels.view(-1).float()
        if self.class_weights is not None:
            class_weights = self.class_weights[labels.long()].to(labels.device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=class_weights)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss

class BinaryClassifier(pl.LightningModule):
    def __init__(self, barlow_twins_model: SelfAttentionBarlowTwinsEmbedder, sample_emb_dim: int, initial_learning_rate: float = 1e-5, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['barlow_twins_model'])
        self.barlow_twins_model = barlow_twins_model.eval()
        for param in self.barlow_twins_model.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(sample_emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # Output layer with a single neuron for binary classification
        ).to(self.device)
        
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.loss_fn = BinaryCrossEntropyLoss(self.class_weights)
        self.train_accuracy = Accuracy(task="binary").to(self.device)
        self.val_accuracy = Accuracy(task="binary").to(self.device)
        self.train_conf_matrix = ConfusionMatrix(task="binary").to(self.device)
        self.val_conf_matrix = ConfusionMatrix(task="binary").to(self.device)
        self.train_precision = Precision(task="binary").to(self.device)
        self.val_precision = Precision(task="binary").to(self.device)
        self.train_recall = Recall(task="binary").to(self.device)
        self.val_recall = Recall(task="binary").to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_emb = self.barlow_twins_model(x.to(self.device))  # Utilizza l'output completo del modello Barlow Twins
        return self.classifier(sample_emb)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        sample_subset1, sample_subset2, labels = batch
        sample_subset1, sample_subset2, labels = sample_subset1.to(self.device), sample_subset2.to(self.device), labels.to(self.device)
        output1 = self(sample_subset1).view(-1)
        output2 = self(sample_subset2).view(-1)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        self.log('train_class_loss', class_loss)
        pred1 = torch.sigmoid(output1) > 0.5
        pred2 = torch.sigmoid(output2) > 0.5
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.train_accuracy(combined_preds, combined_labels)
        self.log('train_accuracy', accuracy)
        precision = self.train_precision(combined_preds, combined_labels)
        self.log('train_precision', precision)
        recall = self.train_recall(combined_preds, combined_labels)
        self.log('train_recall', recall)
        return class_loss

    def validation_step(self, batch, batch_idx: int):
        sample_subset1, sample_subset2, labels = batch
        sample_subset1, sample_subset2, labels = sample_subset1.to(self.device), sample_subset2.to(self.device), labels.to(self.device)
        output1 = self(sample_subset1).view(-1)
        output2 = self(sample_subset2).view(-1)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        self.log('val_class_loss', class_loss)
        pred1 = torch.sigmoid(output1) > 0.5
        pred2 = torch.sigmoid(output2) > 0.5
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.val_accuracy(combined_preds, combined_labels)
        self.log('val_accuracy', accuracy)
        precision = self.val_precision(combined_preds, combined_labels)
        self.log('val_precision', precision)
        recall = self.val_recall(combined_preds, combined_labels)
        self.log('val_recall', recall)
        return class_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.initial_learning_rate, total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]
