"""Training loop for the GNN models (target-agnostic).

Huber (smooth-L1) loss, AdamW optimizer, ReduceLROnPlateau scheduler, gradient
clipping, and checkpoint saving on val_mae improvement. Same loop is used for
hads (H adsorption energy), wf (work function), and magmom targets. Huber is the
shared objective across every trained model (including the CHGNet fine-tune
baseline) so the comparison is on one loss.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)

DEFAULT_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_PATIENCE: int = 100
DEFAULT_MAX_EPOCHS: int = 1000
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_GRAD_CLIP: float = 1.0
# Huber transition point (target units). Below it the loss is quadratic (like
# MSE), above it linear (like MAE), so typical small residuals are fit tightly
# while the hard antiferromagnetic outliers do not dominate. Matches the delta
# CHGNet's Huber criterion uses, so every trained model shares one objective.
HUBER_DELTA: float = 0.1
SCHEDULER_FACTOR: float = 0.5
SCHEDULER_PATIENCE: int = 10
SCHEDULER_MIN_LR: float = 1e-6
LOG_EVERY_N_EPOCHS: int = 25


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch.

    Args:
        model: GNN model
        loader: Training data loader
        optimizer: Optimizer instance
        device: Device string (cuda/cpu)

    Returns:
        Average training Huber loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        predictions = model(batch)
        loss = torch.nn.functional.huber_loss(
            predictions, batch.y, delta=HUBER_DELTA,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=DEFAULT_GRAD_CLIP,
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    assert n_batches > 0, "No training batches processed"
    assert total_loss >= 0, "Loss must be non-negative"
    return total_loss / n_batches


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[list[float], list[float], float, int]:
    """Run inference and collect predictions.

    Args:
        model: Model in eval mode
        loader: Data loader
        device: Device string

    Returns:
        Tuple of (predictions, targets, total_loss, n_batches)
    """
    assert loader is not None, "loader must not be None"
    assert len(device) > 0, "device must not be empty"

    all_predictions: list[float] = []
    all_targets: list[float] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = torch.nn.functional.huber_loss(
                predictions, batch.y, delta=HUBER_DELTA,
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist(),
            )
            all_targets.extend(
                batch.y.cpu().numpy().tolist(),
            )
            total_loss += loss.item()
            n_batches += 1

    return all_predictions, all_targets, total_loss, n_batches


EMPTY_VAL_METRICS: dict[str, float] = {
    "val_loss": float("inf"),
    "val_mae": float("inf"),
    "val_rmse": float("inf"),
    "val_r2": -float("inf"),
}


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Validate model on data loader.

    Args:
        model: GNN model
        loader: Validation data loader
        device: Device string

    Returns:
        Dict with val_loss, val_mae, val_rmse, val_r2
    """
    model.eval()
    preds, targets, total_loss, n_batches = (
        _collect_predictions(model, loader, device)
    )

    if n_batches == 0:
        return dict(EMPTY_VAL_METRICS)

    y_pred = np.array(preds)
    y_true = np.array(targets)

    assert len(y_pred) == len(y_true), "Length mismatch"
    assert len(y_pred) > 0, "No validation samples"

    return {
        "val_loss": total_loss / n_batches,
        "val_mae": mean_absolute_error(y_true, y_pred),
        "val_rmse": root_mean_squared_error(y_true, y_pred),
        "val_r2": r2_score(y_true, y_pred),
    }


def _setup_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
]:
    """Create AdamW optimizer and ReduceLROnPlateau scheduler.

    Args:
        model: Model whose parameters to optimize
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight

    Returns:
        Tuple of (optimizer, scheduler)
    """
    assert learning_rate > 0, "Learning rate must be positive"
    assert weight_decay >= 0, "Weight decay must be non-negative"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )
    return optimizer, scheduler


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_mae: float,
    config: dict[str, Any],
    path: Path,
) -> None:
    """Save model checkpoint to disk.

    Args:
        model: Trained model
        optimizer: Optimizer with current state
        epoch: Current epoch number
        val_mae: Validation MAE at this epoch
        config: Training config dict
        path: Output file path
    """
    assert epoch >= 0, "Epoch must be non-negative"
    assert val_mae >= 0, "MAE must be non-negative"

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_mae": val_mae,
        "config": config,
    }, path)
    logger.info(
        "New best at epoch %d (MAE=%.4f)", epoch, val_mae,
    )


def _extract_train_config(
    config: dict[str, Any],
) -> tuple[str, float, float, int, int, Path, str]:
    """Extract training parameters from config.

    Args:
        config: Training config dict

    Returns:
        Tuple of (device, lr, weight_decay, max_epochs,
            patience, output_dir, checkpoint_name)
    """
    max_epochs = config.get("max_epochs", DEFAULT_MAX_EPOCHS)
    patience = config.get("patience", DEFAULT_PATIENCE)

    assert max_epochs > 0, "max_epochs must be positive"
    assert patience > 0, "patience must be positive"

    checkpoint_name = config.get("checkpoint_name")
    assert checkpoint_name, (
        "config['checkpoint_name'] must be set by the caller"
    )

    return (
        config.get("device", "cpu"),
        config.get("lr", DEFAULT_LR),
        config.get("weight_decay", DEFAULT_WEIGHT_DECAY),
        max_epochs,
        patience,
        Path(config.get("output_dir", "checkpoints")),
        checkpoint_name,
    )


EpochCallback = Optional[
    Callable[[int, float, dict[str, float]], None]
]


class _TrainState:
    """Mutable state for the training loop."""

    def __init__(self, patience: int) -> None:
        """Initialize training state tracker.

        Args:
            patience: Max epochs without improvement
        """
        assert patience > 0, "Patience must be positive"
        self.best_val_mae: float = float("inf")
        self.best_epoch: int = 0
        self.patience_counter: int = 0
        self.patience: int = patience

    def update(self, epoch: int, val_mae: float) -> bool:
        """Update state, return True to stop.

        Args:
            epoch: Current epoch index
            val_mae: Validation MAE this epoch

        Returns:
            True if early stopping triggered
        """
        assert epoch >= 0, "Epoch must be non-negative"
        assert val_mae >= 0, "MAE must be non-negative"

        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
            self.best_epoch = epoch
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.patience


def _step_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, dict[str, float]]:
    """Run one train+validate cycle.

    Args:
        model: Model on target device
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        device: Device string

    Returns:
        Tuple of (train_loss, val_metrics_dict)
    """
    train_loss = train_epoch(
        model, train_loader, optimizer, device,
    )
    val_metrics = validate(model, val_loader, device)

    assert train_loss >= 0, "Train loss must be non-negative"
    return train_loss, val_metrics


def _process_epoch(
    epoch: int,
    state: '_TrainState',
    loss: float,
    metrics: dict[str, float],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    best_path: Path,
    epoch_callback: EpochCallback,
) -> bool:
    """Process one epoch: log, checkpoint, early stop.

    Args:
        epoch: Current epoch index
        state: Training state tracker
        loss: Training loss
        metrics: Validation metrics
        model: Model instance
        optimizer: Optimizer
        config: Training config
        best_path: Checkpoint save path
        epoch_callback: Optional callback

    Returns:
        True if training should stop
    """
    assert epoch >= 0, "epoch must be non-negative"
    assert loss >= 0, "loss must be non-negative"

    val_mae = metrics["val_mae"]
    if epoch_callback is not None:
        epoch_callback(epoch, loss, metrics)
    if epoch % LOG_EVERY_N_EPOCHS == 0:
        logger.info(
            "Epoch %4d | loss=%.4f | mae=%.4f",
            epoch, loss, val_mae,
        )
    should_stop = state.update(epoch, val_mae)
    if val_mae <= state.best_val_mae:
        _save_checkpoint(
            model, optimizer, epoch,
            val_mae, config, best_path,
        )
    if should_stop:
        logger.info(
            "Early stop epoch %d best=%d mae=%.4f",
            epoch, state.best_epoch, state.best_val_mae,
        )
    return should_stop


def _run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    best_path: Path,
    epoch_callback: EpochCallback = None,
) -> tuple[int, float]:
    """Execute epoch loop with early stopping.

    Args:
        model: Model on target device
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training config dict
        best_path: Checkpoint save path
        epoch_callback: Per-epoch callback

    Returns:
        (best_epoch, best_val_mae)
    """
    (device, lr, wd, max_epochs, patience,
     _out_dir, _ckpt) = _extract_train_config(config)
    optimizer, scheduler = _setup_optimizer_and_scheduler(
        model, lr, wd,
    )
    state = _TrainState(patience)

    for epoch in range(max_epochs):
        loss, metrics = _step_epoch(
            model, train_loader, val_loader,
            optimizer, device,
        )
        scheduler.step(metrics["val_mae"])
        stop = _process_epoch(
            epoch, state, loss, metrics,
            model, optimizer, config,
            best_path, epoch_callback,
        )
        if stop:
            break

    return state.best_epoch, state.best_val_mae


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    epoch_callback: EpochCallback = None,
) -> Path:
    """Full training loop with early stopping.

    Args:
        model: GNN model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training config dict (must include checkpoint_name)
        epoch_callback: Called after each epoch

    Returns:
        Path to best saved checkpoint
    """
    (device, learning_rate, _weight_decay,
     max_epochs, patience, output_dir,
     checkpoint_name) = _extract_train_config(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{checkpoint_name}.pt"
    model = model.to(device)

    logger.info(
        "Training: device=%s lr=%.1e epochs=%d patience=%d",
        device, learning_rate, max_epochs, patience,
    )

    best_epoch, best_val_mae = _run_training_loop(
        model, train_loader, val_loader,
        config, best_path, epoch_callback,
    )

    logger.info(
        "Training complete: best_epoch=%d best_mae=%.4f",
        best_epoch, best_val_mae,
    )
    return best_path
