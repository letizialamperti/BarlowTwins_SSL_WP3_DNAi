import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from torch.utils.data import DataLoader

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, logits, labels):
        logits = logits.to(labels.device)

        # Normalize logits by dividing by the number of classes
        logits = logits / self.num_classes

        logits = logits.view(-1, self.num_classes)
        labels = labels.view(-1)

        cum_probs = torch.sigmoid(logits)
        cum_probs = torch.cat([cum_probs, torch.ones_like(cum_probs[:, :1])], dim=1)
        prob = cum_probs[:, :-1] - cum_probs[:, 1:]

        one_hot_labels = torch.zeros_like(prob).scatter(1, labels.unsqueeze(1), 1)

        epsilon = 1e-9
        prob = torch.clamp(prob, min=epsilon, max=1-epsilon)

        if self.class_weights is not None:
            weights = self.class_weights[labels].view(-1, 1)
            loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1) * weights.squeeze()
        else:
            loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1)

        return loss.mean()

class Classifier(pl.LightningModule):
    def __init__(self, barlow_twins_model: SelfAttentionBarlowTwinsEmbedder, sample_repr_dim: int, num_classes: int, initial_learning_rate: float = 1e-5, train_dataloader: DataLoader = None):
        super().__init__()
        self.save_hyperparameters(ignore=['barlow_twins_model', 'train_dataloader'])
        self.barlow_twins_model = barlow_twins_model.eval()
        self.num_classes = num_classes

        self.class_weights = self.calculate_class_weights(train_dataloader)

        for param in self.barlow_twins_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(sample_repr_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

        self.loss_fn = OrdinalCrossEntropyLoss(num_classes, self.class_weights)

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

    def calculate_class_weights(self, train_dataloader: DataLoader):
        if train_dataloader is None:
            return None

        class_counts = torch.zeros(self.num_classes, dtype=torch.float)
        for batch in train_dataloader:
            _, _, labels = batch
            class_counts += torch.bincount(labels, minlength=self.num_classes).float()

        total_count = class_counts.sum()
        class_weights = total_count / (self.num_classes * class_counts)
        return class_weights

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

        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.train_accuracy(combined_preds, combined_labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)

        precision = self.train_precision(combined_preds, combined_labels)
        self.log('train_precision', precision, on_step=False, on_epoch=True)

        recall = self.train_recall(combined_preds, combined_labels)
        self.log('train_recall', recall, on_step=False, on_epoch=True)

        conf_matrix = self.train_conf_matrix(combined_preds, combined_labels)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.log(f'train_conf_matrix_{i}_{j}', conf_matrix[i][j], on_step=False, on_epoch=True)

        return class_loss

    def validation_step(self, batch, batch_idx: int):
        sample_subset1, sample_subset2, labels = batch

        if torch.any(labels >= self.num_classes):
            raise ValueError("Labels out of range")

        output1 = self(sample_subset1)
        output2 = self(sample_subset2)

        labels = labels.to(self.device)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)

        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)
        combined_preds = torch.cat((pred1, pred2), dim=0)
        combined_labels = torch.cat((labels, labels), dim=0)
        accuracy = self.val_accuracy(combined_preds, combined_labels)

        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', class_loss)

        precision = self.val_precision(combined_preds, combined_labels)
        self.log('val_precision', precision, on_step=False, on_epoch=True)

        recall = self.val_recall(combined_preds, combined_labels)
        self.log('val_recall', recall, on_step=False, on_epoch=True)

        conf_matrix = self.val_conf_matrix(combined_preds, combined_labels)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.log(f'val_conf_matrix_{i}_{j}', conf_matrix[i][j], on_step=False, on_epoch=True)

        return class_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate)
        return optimizer
