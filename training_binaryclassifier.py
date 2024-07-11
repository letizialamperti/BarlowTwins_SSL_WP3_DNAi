import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.binary_classifier import BinaryClassifier
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.argparser import get_args, write_config_file
import wandb  # Importa Wandb

# Usa la stessa configurazione
args = get_args()
if args.arg_log:
    write_config_file(args)

pl.seed_everything(args.seed)

samples_dir = Path(args.samples_dir).resolve()

datamodule = BarlowTwinsDataModule(samples_dir=samples_dir,
                                   labels_file=Path(args.labels_file).resolve(), 
                                   sequence_length=args.sequence_length, 
                                   sample_subset_size=args.sample_subset_size,
                                   batch_size=args.batch_size)

# Setup the data module (ensuring that train_dataset is defined)
datamodule.setup(stage='fit')

# Carica il modello Barlow Twins addestrato
barlow_twins_model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint("checkpoints/BT_model-epoch=01-v1.ckpt")

# Crea il classificatore binario con il modello Barlow Twins congelato
model = BinaryClassifier(barlow_twins_model=barlow_twins_model, 
                         sample_repr_dim=args.sample_repr_dim, 
                         initial_learning_rate=args.initial_learning_rate,
                         train_dataset=datamodule.get_train_dataset())

# Checkpoint directory
checkpoint_dir = Path('checkpoints_classifier')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# General checkpoint callback for best model saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',  # Ensure this metric is logged in your model
    dirpath=checkpoint_dir,
    filename='bin_classifier-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)

# Setup logger e trainer
wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir=Path("lightning_logs"), config=args, log_model=False)
trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=10,
    detect_anomaly=False
)

# Inizializzazione Wandb
wandb.init(project='ORDNA_Class', config=args)

# Start training
trainer.fit(model=model, datamodule=datamodule)

# Chiudi Wandb
wandb.finish()
