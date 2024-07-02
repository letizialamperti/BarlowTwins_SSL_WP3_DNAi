import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataset  # Assicurati di avere un dataset per Barlow Twins
from ORDNA.models.classifier import Classifier
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# Impostazioni
CHECKPOINT_PATH = Path('checkpoints_classifier/classifier-epoch=01-val_accuracy=1.00.ckpt')
DATASET = 'sud_corse'
TEST_SAMPLES_DIR = Path(f'/store/sdsc/sd29/letizia/sud_corse_2')  # Directory test set 
SEQUENCE_LENGTH = 300
SAMPLE_SUBSET_SIZE = 500
NUM_CLASSES = 2  # Adjust this based on the number of classes in your classifier
LABELS_FILE = Path('label/labels_binary_sud_corse_test.csv')  # Path to the file containing labels
INITIAL_LEARNING_RATE = 1e-3
BATCH_SIZE = 32
BARLOW_CHECKPOINT_PATH = Path('checkpoints/BT_model-epoch=01.ckpt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path, model_class, datamodule):
    model = model_class.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def test_model(model, dataloader, device):
    model.to(device)
    all_preds = []
    all_labels = []

    for batch in dataloader:
        sample_subset1, sample_subset2, labels = batch
        sample_subset1 = sample_subset1.to(device)
        sample_subset2 = sample_subset2.to(device)
        labels = labels.to(device)

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
    pl.seed_everything(0)

    # Creare il DataLoader per il set di test
    test_dataset = BarlowTwinsDataset(samples_dir=TEST_SAMPLES_DIR,
                                      labels_file=LABELS_FILE, 
                                      sample_subset_size=SAMPLE_SUBSET_SIZE,
                                      sequence_length=SEQUENCE_LENGTH)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=BATCH_SIZE, 
                                 shuffle=False, 
                                 num_workers=12, 
                                 pin_memory=torch.cuda.is_available(), 
                                 drop_last=False)

    # Carica il modello Classifier addestrato
    barlow_twins_model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(BARLOW_CHECKPOINT_PATH)
    model = Classifier.load_from_checkpoint(CHECKPOINT_PATH, 
                                            barlow_twins_model=barlow_twins_model, 
                                            sample_repr_dim=128,  # Sample representation dimension
                                            num_classes=NUM_CLASSES, 
                                            initial_learning_rate=INITIAL_LEARNING_RATE)
    
    model.eval()

    # Esegui il test
    preds, labels = test_model(model, test_dataloader, device)

    # Calcola e stampa i risultati
    if NUM_CLASSES > 2:
        acc = accuracy_score(labels, preds)
        print(f"Test Accuracy: {acc}")
        print("Classification Report:\n", classification_report(labels, preds))
        print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    else:
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        print(f"Test Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:\n", confusion_matrix(labels, preds))
