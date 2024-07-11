import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def calculate_class_weights(dataset, num_classes):
    labels = []
    for _, _, label in dataset:
        labels.append(label)
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    return class_weights

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, logits, labels):
        logits = logits.to(labels.device)
        logits = logits.view(-1, self.num_classes)
        labels = labels.view(-1)
        cum_probs = torch.sigmoid(logits)
        cum_probs = torch.cat([cum_probs, torch.ones_like(cum_probs[:, :1])], dim=1)
        prob = cum_probs[:, :-1] - cum_probs[:, 1:]
        one_hot_labels = torch.zeros_like(prob).scatter(1, labels.unsqueeze(1), 1)
        epsilon = 1e-9
        prob = torch.clamp(prob, min=epsilon, max=1-epsilon)
        if self.class_weights is not None:
            class_weights = self.class_weights[labels].view(-1, 1)
            loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1) * class_weights
        else:
            loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1)
        return loss.mean()

class BaselineClassifier(pl.LightningModule):
    def __init__(self, input_dim: int, num_classes: int, initial_learning_rate: float = 1e-5, train_dataset=None):
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataset'])
        self.num_classes = num_classes

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.class_weights = calculate_class_weights(train_dataset, num_classes).to(self.device) if train_dataset is not None else None
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits.squeeze(dim=1)

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
