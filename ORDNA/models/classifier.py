import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix

class Classifier(pl.LightningModule):
    def __init__(self, sample_repr_dim: int, num_classes: int, initial_learning_rate: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes

        # Classifier adjusted for dynamic number of classes
        self.classifier = nn.Sequential(
            nn.Linear(sample_repr_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1 if num_classes == 2 else num_classes)  # Output 1 if binary classification
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()

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

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(dim=1)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        sample_repr, labels = batch

        output = self(sample_repr)
        
        # Classification loss
        if self.num_classes > 2:
            class_loss = self.loss_fn(output, labels)
        else:
            labels = labels.float()
            class_loss = self.loss_fn(output, labels)
        self.log('class_loss', class_loss)

        # Accuracy calculation
        if self.num_classes > 2:
            pred = torch.argmax(output, dim=1)
        else:
            pred = (torch.sigmoid(output) > 0.5).long()
            
        accuracy = self.train_accuracy(pred, labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)

        # Compute confusion matrix for additional metrics
        conf_matrix = self.train_conf_matrix(pred, labels)
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        self.log('train_FP', FP, on_step=False, on_epoch=True)
        self.log('train_FN', FN, on_step=False, on_epoch=True)

        return class_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int):
        sample_repr, labels = batch

        output = self(sample_repr)

        # Classification loss
        if self.num_classes > 2:
            class_loss = self.loss_fn(output, labels)
            pred = torch.argmax(output, dim=1)
        else:
            labels = labels.float()
            class_loss = self.loss_fn(output, labels)
            pred = (torch.sigmoid(output) > 0.5).long()

        # Accuracy calculation
        accuracy = self.val_accuracy(pred, labels)
    
        # Log the combined accuracy
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', class_loss)

        return class_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.initial_learning_rate)
        return optimizer
