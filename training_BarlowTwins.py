import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.argparser import get_args, write_config_file

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

model = SelfAttentionBarlowTwinsEmbedder(token_emb_dim=args.token_emb_dim, 
                                         seq_len=args.sequence_length,
                                         sample_repr_dim=args.sample_repr_dim, 
                                         sample_emb_dim=args.sample_emb_dim, 
                                         lmbda=args.barlow_twins_lambda, 
                                         initial_learning_rate=args.initial_learning_rate)

# General checkpoint callback for best model saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    dirpath='checkpoints',
    filename='model-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)

# Setup logger and trainer
wandb_logger = WandbLogger(project='ORDNA_Class', save_dir=Path("lightning_logs"), config=args, log_model=False)
trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=10,
    detect_anomaly=False
)

# Start training
trainer.fit(model=model, datamodule=datamodule)
