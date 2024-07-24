import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.classifier_from_scratch import ClassifierFromScratch
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

print("Initializing classifier from scratch model...")
# Crea il classificatore da zero
model = ClassifierFromScratch(token_emb_dim=args.token_emb_dim, 
                              seq_len=args.sequence_length, 
                              repr_dim=args.sample_emb_dim, 
                              num_classes=args.num_classes, 
                              initial_learning_rate=args.initial_learning_rate,
                              class_weights=class_weights)

print("Setting up checkpoint directory...")
# Checkpoint directory
checkpoint_dir = Path('checkpoints_classifier_from_scratch')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Initializing checkpoint callback...")
# General checkpoint callback for best model saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',  # Ensure this metric is logged in your model
    dirpath=checkpoint_dir,
    filename='scratch_classifier-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)

# Custom Early stopping callback
class CustomEarlyStopping(pl.Callback):
    def __init__(self, monitor: str, patience: int, mode: str = 'min', min_delta: float = 0.0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.wait = 0
        self.best_score = None
        self.stopped_epoch = 0
        self.stop_training = False

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        if self.best_score is None:
            self.best_score = current
            return

        if self.mode == 'min' and current < self.best_score - self.min_delta:
            self.best_score = current
            self.wait = 0
        elif self.mode == 'max' and current > self.best_score + self.min_delta:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True
            self.stop_training = True
            print(f"Early stopping triggered at epoch {self.stopped_epoch} with best {self.monitor}: {self.best_score}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.stop_training:
            print(f"Stopping training at step {trainer.global_step + 1} due to early stopping.")
            trainer.should_stop = True
            trainer.train_loop.run = False  # Stops training immediately

# Callback for validation on each step
class ValidationOnStepCallback(pl.Callback):
    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_step = trainer.global_step + 1
        if current_step % self.n_steps == 0:
            print(f"[DEBUG] Running validation at step {current_step}")
            # Esegui manualmente la validazione
            pl_module.eval()
            val_dataloader = trainer.datamodule.val_dataloader()
            val_class_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    sample_subset1, sample_subset2, labels = batch
                    sample_subset1, sample_subset2, labels = sample_subset1.to(pl_module.device), sample_subset2.to(pl_module.device), labels.to(pl_module.device)
                    output1 = pl_module(sample_subset1)
                    output2 = pl_module(sample_subset2)
                    val_class_loss += pl_module.loss_fn(output1, labels).item()
                    val_class_loss += pl_module.loss_fn(output2, labels).item()
                    pred1 = torch.argmax(output1, dim=1)
                    pred2 = torch.argmax(output2, dim=1)
                    correct += (pred1 == labels).sum().item()
                    correct += (pred2 == labels).sum().item()
                    total += labels.size(0) * 2  # Due predizioni per batch
            val_class_loss /= len(val_dataloader)
            val_accuracy = correct / total
            print(f"[DEBUG] Validation at step {current_step}: val_class_loss = {val_class_loss}, val_accuracy = {val_accuracy}")
            pl_module.train()

print("Setting up Wandb logger...")
# Setup logger e trainer
wandb_logger = WandbLogger(project='ORDNA_Class_july_scratch', save_dir=Path("lightning_logs"), config=args, log_model=False)

# Inizializzazione Wandb
print("Initializing Wandb run...")
wandb_run = wandb.init(project='ORDNA_Class_july_scratch', config=args)

# Print Wandb run URL
print(f"Wandb run URL: {wandb_run.url}")

print("Initializing trainer...")

# Parametri del dataset e batch size
N = len(datamodule.train_dataloader().dataset)  # Numero di campioni di addestramento
B = args.batch_size  # Batch size

# Calcolare il numero totale di batch per epoca
num_batches_per_epoch = N // B
print(f"Number of batches per epoch: {num_batches_per_epoch}")

# Scegliere n_steps come il 10% dei batch per epoca
n_steps = num_batches_per_epoch // 30
print(f"Validation will run every {n_steps} steps")

# Inizializza i callback
validation_callback = ValidationOnStepCallback(n_steps=n_steps)
early_stopping_callback = CustomEarlyStopping(monitor='val_accuracy', patience=3, mode='max')

trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, validation_callback, early_stopping_callback],
    log_every_n_steps=10,
    detect_anomaly=False
)

# Ensure the callbacks are correctly passed
print(f"Trainer callbacks: {trainer.callbacks}")

# Start training
print("Starting training...")
trainer.fit(model=model, datamodule=datamodule)

# Check if early stopping was triggered
if any(callback.stop_training for callback in trainer.callbacks if isinstance(callback, CustomEarlyStopping)):
    stopped_epoch = next(callback.stopped_epoch for callback in trainer.callbacks if isinstance(callback, CustomEarlyStopping))
    print(f"Early stopping was triggered at epoch {stopped_epoch}.")

# Chiudi Wandb
print("Finishing Wandb run...")
wandb.finish()
