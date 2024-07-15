import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.classifier import Classifier
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.argparser import get_args, write_config_file
import pandas as pd

def calculate_class_weights_from_csv(labels_file: Path, num_classes: int):
    print("Calculating class weights from CSV...")
    labels_df = pd.read_csv(labels_file)
    labels = labels_df['label'].values
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    print(f"Class weights: {class_weights}")
    return class_weights

# Controllo se la GPU è disponibile
if torch.cuda.is_available():
    print(f"GPU is available. Device: {torch.cuda.get_device_name(0)}")
else:
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
class_weights = calculate_class_weights_from_csv(Path(args.labels_file).resolve(), num_classes=args.num_classes)

print("Loading Barlow Twins model...")
# Carica il modello Barlow Twins addestrato
barlow_twins_model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint("checkpoints/BT_model-epoch=01-v1.ckpt")

print("Initializing classifier model...")
# Crea il classificatore con il modello Barlow Twins congelato
model = Classifier(barlow_twins_model=barlow_twins_model, 
                   sample_repr_dim=args.sample_repr_dim, 
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
    monitor='val_loss',  # Monitor validation loss
    patience=10,  # Number of validation steps with no improvement after which training will be stopped
    mode='min',
    verbose=True,
    check_on_train_epoch_end=False  # Check on validation steps
)

print("Setting up Wandb logger...")
# Setup logger e trainer
wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir=str(Path("lightning_logs")), config=args, log_model=False)

print("Initializing trainer...")
trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=10,
    detect_anomaly=False
)

# Debug: Print trainer info
print(f"Trainer initialized with logger: {trainer.logger}")

# Start training
print("Starting training...")
trainer.fit(model=model, datamodule=datamodule)
