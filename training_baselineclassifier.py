import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.baseline_classifier import BaselineClassifier  # Assicurati che il nome e il percorso siano corretti
from ORDNA.utils.argparser import get_args, write_config_file

# Usa la stessa configurazione
args = get_args()
if args.arg_log:
    write_config_file(args)

pl.seed_everything(args.seed)

print("Setting up data module...")
samples_dir = Path(args.samples_dir).resolve()

datamodule = BarlowTwinsDataModule(samples_dir=samples_dir,
                                   labels_file=Path(args.labels_file).resolve(), 
                                   sequence_length=args.sequence_length, 
                                   sample_subset_size=args.sample_subset_size,
                                   batch_size=args.batch_size)

# Setup the data module (ensuring that train_dataset is defined)
datamodule.setup(stage='fit')
print("Data module setup complete.")

# Calcola l'input_dim basato sui dati
input_dim = args.sequence_length * args.sample_subset_size
print(f"Input dimension: {input_dim}")

# Crea il nuovo classificatore
model = BaselineClassifier(input_dim=input_dim, 
                           num_classes=args.num_classes, 
                           initial_learning_rate=args.initial_learning_rate,
                           train_dataset=datamodule.get_train_dataset())

print("Model created.")

# Checkpoint directory
checkpoint_dir = Path('checkpoints_classifier')
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint directory: {checkpoint_dir}")

# General checkpoint callback for best model saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',  # Ensure this metric is logged in your model
    dirpath=checkpoint_dir,
    filename='classifier-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)

# Setup logger e trainer
wandb_logger = WandbLogger(project='ORDNA_Class', save_dir=Path("lightning_logs"), config=args, log_model=False)
print("Wandb logger setup complete.")

trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=10,
    detect_anomaly=False
)

print("Trainer setup complete.")

# Start training
trainer.fit(model=model, datamodule=datamodule)
print("Training started.")
