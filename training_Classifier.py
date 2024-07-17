import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.classifier import Classifier
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.argparser import get_args, write_config_file
import wandb

# Funzione per calcolare i pesi delle classi
def calculate_class_weights_from_csv(labels_file: Path, num_classes: int) -> torch.Tensor:
    import pandas as pd
    labels = pd.read_csv(labels_file)
    label_counts = labels['protection'].value_counts().sort_index()
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    return torch.tensor(class_weights.values, dtype=torch.float)

# Controllo se la GPU è disponibile
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")

# Usa la stessa configurazione
args = get_args()
if args.arg_log:
    write_config_file(args)

print("Setting random seed...")
pl.seed_everything(args.seed)

samples_dir = Path(args.samples_dir).resolve()

print("Initializing data module...")
datamodule = BarlowTwinsDataModule(samples_dir=samples_dir,
                                   labels_file=Path(args.labels_file).resolve(), 
                                   sequence_length=args.sequence_length, 
                                   sample_subset_size=args.sample_subset_size,
                                   batch_size=args.batch_size)

print("Setting up data module...")
datamodule.setup(stage='fit')  # Ensure train_dataset is defined

print("Calculating class weights from CSV...")
class_weights = calculate_class_weights_from_csv(Path(args.labels_file).resolve(), args.num_classes)
print(f"Class weights: {class_weights}")

print("Loading Barlow Twins model...")
# Carica il modello Barlow Twins addestrato
barlow_twins_model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint("checkpoints/BT_sud_cose_1_dataset-epoch=00.ckpt").to(device)

print("Initializing classifier model...")
# Crea il classificatore con il modello Barlow Twins congelato
sample_emb_dim = args.sample_emb_dim  # Assicurati che questo sia corretto
model = Classifier(barlow_twins_model=barlow_twins_model, 
                   sample_emb_dim=sample_emb_dim, 
                   num_classes=args.num_classes, 
                   initial_learning_rate=args.initial_learning_rate,
                   class_weights=class_weights)

print("Setting up checkpoint directory...")
# Checkpoint directory
checkpoint_dir = Path('checkpoints_classifier')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Initializing checkpoint callback...")
# General checkpoint callback for best model saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',  # Ensure this metric is logged in your model
    dirpath=checkpoint_dir,
    filename='corse_classifier-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)

print("Initializing early stopping callback...")
# Early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_class_loss',  # Monitor the correct validation loss
    patience=10,  # Number of validation steps with no improvement after which training will be stopped
    mode='min',
    verbose=True,
    check_on_train_epoch_end=False  # Check on validation steps
)

# Callback for validation on each step
class ValidationOnStepCallback(pl.Callback):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def on_batch_end(self, trainer, pl_module):
        if (trainer.global_step + 1) % self.n_steps == 0:
            print(f"Running validation at step {trainer.global_step + 1}")
            val_outputs = trainer.validate(datamodule=trainer.datamodule, verbose=False)
            for output in val_outputs:
                for key, value in output.items():
                    print(f"Logging {key} with value {value}")
                    pl_module.log(key, value, prog_bar=True, logger=True)
            # Manually call `log` for each validation metric
            for metric in ['val_class_loss', 'val_accuracy', 'val_precision', 'val_recall']:
                metric_value = trainer.callback_metrics.get(metric)
                if metric_value is not None:
                    print(f"Logging metric {metric} with value {metric_value}")
                    pl_module.log(metric, metric_value, prog_bar=True, logger=True)

print("Setting up Wandb logger...")
# Setup logger e trainer
wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir=Path("lightning_logs"), config=args, log_model=False)

# Inizializzazione Wandb
print("Initializing Wandb run...")
wandb_run = wandb.init(project='ORDNA_Class_july', config=args)

# Print Wandb run URL
print(f"Wandb run URL: {wandb_run.url}")

print("Initializing trainer...")

# Parametri del dataset e batch size
N = len(datamodule.train_dataloader().dataset)  # Numero di campioni di addestramento
B = args.batch_size  # Batch size

# Calcolare il numero totale di batch per epoca
num_batches_per_epoch = N // B

# Scegliere n_steps come il 10% dei batch per epoca
n_steps = num_batches_per_epoch // 10

# Debug prints for verification
print(f"Number of batches per epoch: {num_batches_per_epoch}")
print(f"Validation every {n_steps} steps")

trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stopping_callback, ValidationOnStepCallback(n_steps=n_steps)],
    log_every_n_steps=10,
    detect_anomaly=False
)

# Start training
print("Starting training...")
trainer.fit(model=model, datamodule=datamodule)

# Chiudi Wandb
print("Finishing Wandb run...")
wandb.finish()
