import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataset  # Assicurati di avere un dataset per Barlow Twins
from ORDNA.models.classifier import Classifier
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.argparser import get_args, write_config_file
import argparse

def load_checkpoint(checkpoint_path, model_class, datamodule):
    model = model_class.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def test_model(model, dataloader):
    all_preds = []
    all_labels = []

    for batch in dataloader:
        sample_subset1, sample_subset2, labels = batch
        with torch.no_grad():
            output1 = model(sample_subset1)
            output2 = model(sample_subset2)
        
        if model.num_classes > 2:
            pred1 = torch.argmax(output1, dim=1)
            pred2 = torch.argmax(output2, dim=1)
        else:
            pred1 = (torch.sigmoid(output1) > 0.5).long()
            pred2 = (torch.sigmoid(output2) > 0.5).long()
        
        all_preds.extend(pred1.cpu().numpy())
        all_preds.extend(pred2.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

if __name__ == "__main__":
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', required=True, type=str, help='Directory of sample files')
    parser.add_argument('--labels_file', required=True, type=str, help='Labels file path')
    parser.add_argument('--sequence_length', required=True, type=int, help='Sequence length')
    parser.add_argument('--num_classes', required=True, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--sample_subset_size', default=32, type=int, help='Sample subset size')
    parser.add_argument('--token_emb_dim', default=128, type=int, help='Token embedding dimension')
    parser.add_argument('--sample_repr_dim', default=128, type=int, help='Sample representation dimension')
    parser.add_argument('--initial_learning_rate', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--test_samples_dir', required=True, type=str, help='Directory of test sample files')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to the classifier checkpoint')
    parser.add_argument('--barlow_checkpoint_path', required=True, type=str, help='Path to the Barlow Twins checkpoint')

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    samples_dir = Path(args.samples_dir).resolve()
    test_samples_dir = Path(args.test_samples_dir).resolve()
    labels_file = Path(args.labels_file).resolve()

    # Creare il DataLoader per il set di test
    test_dataset = BarlowTwinsDataset(samples_dir=test_samples_dir,
                                      labels_file=labels_file, 
                                      sample_subset_size=args.sample_subset_size,
                                      sequence_length=args.sequence_length)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=False, 
                                 num_workers=12, 
                                 pin_memory=torch.cuda.is_available(), 
                                 drop_last=False)

    # Carica il modello Classifier addestrato
    barlow_twins_model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(args.barlow_checkpoint_path)
    model = Classifier.load_from_checkpoint(args.checkpoint_path, 
                                            barlow_twins_model=barlow_twins_model, 
                                            sample_repr_dim=args.sample_repr_dim, 
                                            num_classes=args.num_classes, 
                                            initial_learning_rate=args.initial_learning_rate)
    
    model.eval()

    # Esegui il test
    preds, labels = test_model(model, test_dataloader)

    # Calcola e stampa i risultati
    if args.num_classes > 2:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        acc = accuracy_score(labels, preds)
        print(f"Test Accuracy: {acc}")
        print("Classification Report:\n", classification_report(labels, preds))
        print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    else:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        print(f"Test Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:\n", confusion_matrix(labels, preds))
