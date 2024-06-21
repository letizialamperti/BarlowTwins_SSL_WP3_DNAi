import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from ORDNA.data.triplets_datamodule import TripletsDataModule
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.triplets import SelfAttentionTripletsEmbedder
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.argparser import get_args, write_config_file


# Get current date and time
current_datetime = "Letizia_01"

class DynamicWeightCallback(pl.Callback):
    def __init__(self, weight_increase_epochs, max_weight=1.0):
        super().__init__()
        self.weight_increase_epochs = weight_increase_epochs
        self.max_weight = max_weight

    def on_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch in self.weight_increase_epochs:
            new_weight = self.weight_increase_epochs[current_epoch]
            new_weight = min(new_weight, self.max_weight)  # Ensure weight does not exceed max
            pl_module.classification_loss_weight = new_weight
            trainer.logger.log_metrics({'classification_loss_weight': new_weight}, step=current_epoch)

class PreClassifierCheckpointCallback(pl.Callback):
    def __init__(self, epoch_to_save: int, dirpath: str, filename: str):
        super().__init__()
        self.epoch_to_save = epoch_to_save
        self.dirpath = dirpath
        self.filename = filename
    
    def on_epoch_end(self, trainer, pl_module):
        # Save checkpoint at the specified epoch
        if trainer.current_epoch == self.epoch_to_save:
            ckpt_path = f"{self.dirpath}/{self.filename.format(epoch=trainer.current_epoch)}.ckpt"
            trainer.save_checkpoint(ckpt_path)
            print(f"Checkpoint saved at epoch {self.epoch_to_save}")

args = get_args()
if args.arg_log:
    write_config_file(args)

pl.seed_everything(args.seed)

samples_dir = Path(args.samples_dir).resolve()

if args.embedder_type == 'barlow_twins':
    datamodule = BarlowTwinsDataModule(samples_dir=samples_dir,
                                       labels_file=Path(args.labels_file).resolve(), 
                                       sequence_length=args.sequence_length, 
                                       sample_subset_size=args.sample_subset_size,
                                       batch_size=args.batch_size)

    model = SelfAttentionBarlowTwinsEmbedder(token_emb_dim=args.token_emb_dim, 
                                             seq_len=args.sequence_length,
                                             sample_repr_dim=args.sample_repr_dim, 
                                             sample_emb_dim=args.sample_emb_dim, 
                                             num_classes=args.num_classes,
                                             lmbda=args.barlow_twins_lambda, 
                                             initial_learning_rate=args.initial_learning_rate)
    
elif args.embedder_type == 'triplets':
    datamodule = TripletsDataModule(samples_dir=samples_dir,
                                    sequence_length=args.sequence_length, 
                                    sample_subset_size=args.sample_subset_size,
                                    batch_size=args.batch_size)

    model = SelfAttentionTripletsEmbedder(token_emb_dim=args.token_emb_dim, 
                                          seq_len=args.sequence_length, 
                                          sample_repr_dim=args.sample_repr_dim,
                                          sample_emb_dim=args.sample_emb_dim, 
                                          triplet_margin=args.triplet_margin, 
                                          l2_reg_max_weight=args.l2_reg_max_weight, 
                                          initial_learning_rate=args.initial_learning_rate)
else:
    raise Exception("Unknown embedder type:", args.embedder_type)

# Define weight adjustment plan for classification loss weight
weight_schedule = {0: 0, 1: 1}  # epoch: value for classification loss weight (between 0 and 1)
dynamic_weight_callback = DynamicWeightCallback(weight_increase_epochs=weight_schedule)

# Pre-classifier checkpoint callback
pre_classifier_checkpoint_callback = PreClassifierCheckpointCallback(
    epoch_to_save=1,  # Specify the epoch before the classifier is activated
    dirpath="pre_classifier_checkpoints",
    filename="pre_classifier-{epoch:02d}"
)

# General checkpoint callback for best model saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    dirpath='checkpoints',
    filename='model-{current_datetime}-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)


# Setup logger and trainer
wandb_logger = WandbLogger(project='ORDNA_Class', save_dir=Path("lightning_logs"), config=args, log_model=False)  # Turn off model logging to save space
trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[dynamic_weight_callback, pre_classifier_checkpoint_callback, checkpoint_callback],  # Add your callbacks here
    log_every_n_steps=10,  # Reduce logging frequency
    detect_anomaly=False
)

# Start training
trainer.fit(model=model, datamodule=datamodule)
