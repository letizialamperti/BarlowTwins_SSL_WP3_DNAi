import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall
from ORDNA.models.representation_module import SelfAttentionRepresentationModule

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
            class_weights = self.class_weights[labels].view(-1, 1).to(labels.device)
            loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1) * class_weights
        else:
            loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1)
        return loss.mean()

class ClassifierFromScratch(pl.LightningModule):
    def __init__(self, token_emb_dim: int, seq_len: int, repr_dim: int, num_classes: int, initial_learning_rate: float = 1e-5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Representation module
        self.repr_module = SelfAttentionRepresentationModule(token_emb_dim=token_emb_dim,
                                                             seq_len=seq_len,
                                                             repr_dim=repr_dim)
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        ).to(self.device)
        
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.loss_fn = OrdinalCrossEntropyLoss(num_classes, self.class_weights)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_repr = self.repr_module(x.to(self.device))  # Utilizza l'output della rappresentazione
        return self.classifier(sample_repr)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        sample_subset1, sample_subset2, labels = batch
        sample_subset1, sample_subset2, labels = sample_subset1.to(self.device), sample_subset2.to(self.device), labels.to(self.device)
        output1 = self(sample_subset1)
        output2 = self(sample_subset2)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        self.log('train_class_loss', class_loss)
        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)
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
        output1 = self(sample_subset1)
        output2 = self(sample_subset2)
        class_loss = self.loss_fn(output1, labels) + self.loss_fn(output2, labels)
        self.log('val_class_loss', class_loss)
        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)
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
