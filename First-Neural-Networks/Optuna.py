import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import optuna
from optuna.trial import TrialState
# If using PyTorch Lightning:
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl
import torchmetrics

class LitModel(pl.LightningModule):
    def __init__(self, hidden_size, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
from optuna.integration import PyTorchLightningPruningCallback

def lightning_objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Data preparation (MNIST example)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Lightning model
    model = LitModel(hidden_size=hidden_size, learning_rate=learning_rate)

    # Setup PyTorch Lightning Trainer with Optuna Pruning Callback
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto", # Use GPU if available, otherwise CPU
        devices=1,
        logger=False, # Disable default logger to avoid clutter
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
    )

    # Fit the model
    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_accuracy"].item() # Return the metric to optimize

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1))

    study.optimize(lightning_objective, n_trials=50, timeout=600)

    print("Best trial (PyTorch Lightning):")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")