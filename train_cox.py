"""
Cox model training and evaluation.

Author: Ahmed H. Shahin
Date: 31/08/2023
"""

import os
from argparse import Namespace
from time import time
from typing import Any, Tuple

import numpy as np
import torch

from eval_utils import cindex, get_median_survival_cox, mae, rae
from losses import cox_loss
from utils import collect_batch_stats, get_data, get_grad_norm
from utils import (
    initialize,
    initialize_model_and_optimizer,
    log_metrics,
    parse_args,
    prepare_batch,
    prepare_epoch_stats,
    save_checkpoint,
)


def parse_args_cox() -> Namespace:
    """
    Get arguments from command line. Use the base parser in utils.py and add
      Cox-specific arguments, if any.

    Returns:
        args: Namespace object containing arguments.
    """
    parser = parse_args()
    return parser.parse_args()


def train_one_epoch(train_args, model, loader, optim, epoch, writer):
    """
    Train the model for one epoch.

    Args:
        train_args: The command-line arguments or a configuration object.
        model: The PyTorch model to be trained.
        loader: The DataLoader for the training set.
        optim: The optimizer.
        epoch: The current epoch number.
        writer: The TensorBoard writer object for logging.

    Returns:
        epoch_loss: The average loss for this epoch.
        cindex: The concordance index for the training set.
        mae_nc: The mean absolute error for uncensored data.
        preds: The model's predictions.
        events: The event occurrences.
        times: The time-to-event data.
    """
    model.train()
    device = train_args.device
    epoch_loss = 0.0
    all_preds, all_events, all_times = [], [], []
    for batch_idx, sample in enumerate(loader):
        img, clinical_data, time_to_event, event = prepare_batch(sample, device)

        optim.zero_grad()
        risk = model(img, clinical_data)

        loss = cox_loss(risk, event, time_to_event)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 100.0, error_if_nonfinite=True
        )
        optim.step()
        grad_norm = get_grad_norm(model)
        writer.add_scalar("grad_norm", grad_norm, batch_idx + epoch * len(loader))
        writer.add_scalar("loss_per_iteration", loss, batch_idx + epoch * len(loader))
        epoch_loss += loss.item() * event.sum().item()

        collect_batch_stats(
            all_preds, all_events, all_times, risk, event, time_to_event
        )

    preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)
    preds_times = get_median_survival_cox(preds, events, times, preds)
    preds_times = np.minimum(preds_times, train_args.tmax)
    cindex_score = cindex(times, events, preds)
    mae_nc = mae(preds_times, times, events, "uncens")
    return (
        epoch_loss / loader.dataset.event.sum(),
        cindex_score,
        mae_nc,
        preds,
        events,
        times,
    )


def validate_one_epoch(val_args, model, loader, tr_preds, tr_events, tr_times):
    """
    Validate the model for one epoch.

    Args:
        val_args: The command-line arguments or a configuration object.
        model: The PyTorch model to be validated.
        loader: The DataLoader for the validation set.
        tr_preds: The training set predictions.
        tr_events: The training set event occurrences.
        tr_times: The training set time-to-event data.

    Returns:
        epoch_loss: The average loss for this epoch.
        cindex: The concordance index for the validation set.
        rae_nc: The relative absolute error for uncensored data.
        rae_c: The relative absolute error for censored data.
        mae_nc: The mean absolute error for uncensored data.
        mae_c: The mean absolute error for censored data.
    """
    model.eval()
    device = val_args.device
    epoch_loss = 0.0
    all_preds, all_events, all_times = [], [], []
    with torch.no_grad():
        for sample in loader:
            img, clinical_data, time_to_event, event = prepare_batch(sample, device)

            risk = model(img, clinical_data)
            loss = cox_loss(risk, event, time_to_event)
            epoch_loss += loss.item() * event.sum().item()

            collect_batch_stats(
                all_preds, all_events, all_times, risk, event, time_to_event
            )

        epoch_loss /= loader.dataset.event.sum()
        preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)

        cindex_score = cindex(times, events, preds)
        preds_times = get_median_survival_cox(tr_preds, tr_events, tr_times, preds)
        preds_times = np.minimum(preds_times, val_args.tmax)
        mae_nc = mae(preds_times, times, events, "uncens")
        mae_c = mae(preds_times, times, events, "cens")
        rae_nc = rae(preds_times, times, events, "uncens")
        rae_c = rae(preds_times, times, events, "cens")
        return epoch_loss, cindex_score, rae_nc, rae_c, mae_nc, mae_c


def load_checkpoint(ckpt_path, device):
    """Load a checkpoint if provided and return relevant state variables."""

    # Initialize default values
    ckpt = None
    epoch = 0
    best_epoch = 0
    best_cindex = -float("inf")  # Initialize to negative infinity

    # Try to load the checkpoint if provided
    if ckpt_path:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            print(f"Checkpoint loaded from {ckpt_path}")
        except FileNotFoundError:
            print(f"No checkpoint found at {ckpt_path}")

    return ckpt, epoch, best_epoch, best_cindex


def get_preds(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get model predictions, events, and times for a given data loader.

    Args:
        model: The PyTorch model to be evaluated.
        loader: DataLoader for the dataset.
        device: Device to use for evaluation.

    Returns:
        preds: Predictions from the model.
        events: Event occurrences.
        times: Time to events.
        pid_sids: Patient IDs
    """
    model.eval()

    all_preds, all_events, all_times, all_pid_sids = [], [], [], []

    with torch.no_grad():
        for sample in loader:
            img, clinical_data, time_to_event, event = prepare_batch(sample, device)

            risk = model(img, clinical_data)

            collect_batch_stats(
                all_preds, all_events, all_times, risk, event, time_to_event
            )
            all_pid_sids.extend(sample["pid_sid"])

    preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)
    pid_sids = np.array(all_pid_sids).reshape(-1)
    return preds, events, times, pid_sids


def train_and_validate_epoch(
    train_val_args,
    output_dir,
    writer,
    epoch,
    model,
    optim,
    sched,
    train_loader,
    val_loader,
):
    """
    Train and validate the model for one epoch.

    Args:
        train_val_args: The command-line arguments or a configuration object.
        output_dir: The directory where outputs will be saved.
        writer: The TensorBoard writer object for logging.
        epoch: The current epoch number.
        model: The PyTorch model to be trained.
        optim: The optimizer.
        sched: The learning rate scheduler.
        train_loader: The DataLoader for the training set.
        val_loader: The DataLoader for the validation set.

    Returns:
        cindex: The concordance index for the validation set.
    """
    start_time = time()
    (
        train_loss,
        cindex_train,
        mae_nc_train,
        train_preds,
        train_events,
        train_times,
    ) = train_one_epoch(train_val_args, model, train_loader, optim, epoch, writer)

    val_loss, cindex_score, rae_nc, rae_c, mae_nc, mae_c = validate_one_epoch(
        train_val_args, model, val_loader, train_preds, train_events, train_times
    )
    sched.step()
    log_metrics(
        train_val_args.epochs,
        writer,
        epoch,
        start_time,
        train_loss,
        cindex_train,
        mae_nc_train,
        val_loss,
        cindex_score,
        rae_nc,
        rae_c,
        mae_nc,
        mae_c,
    )
    save_checkpoint(
        model,
        optim,
        epoch,
        cindex_score,
        suffix="last",
        output_dir=output_dir,
        metric="cindex",
    )
    return cindex_score


def test(
    test_args: Any,
    output_dir: str,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
) -> None:
    """Test the model and print the results.

    Args:
        args: Command-line arguments.
        output_dir: Directory where the best model checkpoint is saved.
        model: The PyTorch model to be tested.
        train_loader: DataLoader for the training set.
    """
    # Load the best model
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "checkpoint_best.pth"))["model"]
    )
    model.eval()

    # Get predictions and ground truth from the training set
    train_preds, train_events, train_times, _ = get_preds(
        model, train_loader, test_args.device
    )

    # Clear memory
    del train_loader

    # Get the test data loader
    test_loader = get_data(test_args, test=True)

    # Validate the model on the test set
    test_loss, cindex_score, rae_nc, rae_c, mae_nc, mae_c = validate_one_epoch(
        test_args, model, test_loader, train_preds, train_events, train_times
    )

    # Log the test results
    print("Test results:")
    print(
        f"Test loss: {test_loss:.4f} | C-index: {cindex_score:.4f} |\
              MAE_nc: {mae_nc:.4f} | MAE_c: {mae_c:.4f} |\
                  RAE_nc: {rae_nc:.4f} | RAE_c: {rae_c:.4f}"
    )


def main(main_args):
    """
    Main function for training and testing the model.
    """
    output_dir, writer = initialize(main_args)

    main_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {main_args.device}")

    # load checkpoint, if provided
    ckpt, epoch, best_epoch, best_cindex = load_checkpoint(
        main_args.checkpoint_path, main_args.device
    )

    # model and optimizer
    model, optim, sched = initialize_model_and_optimizer(main_args, ckpt)

    # load the data
    train_loader, val_loader = get_data(main_args)

    for epoch in range(epoch, main_args.epochs):
        if ((epoch - best_epoch) > main_args.patience) and (
            epoch > main_args.min_epochs
        ):
            break

        cindex_score = train_and_validate_epoch(
            main_args,
            output_dir,
            writer,
            epoch,
            model,
            optim,
            sched,
            train_loader,
            val_loader,
        )
        if cindex_score > best_cindex:
            print("Saving best model...")
            best_cindex = cindex_score
            best_epoch = epoch
            save_checkpoint(
                model,
                optim,
                epoch,
                best_cindex,
                suffix="best",
                output_dir=output_dir,
                metric="cindex",
            )
        print()
    print(f"Best epoch: {best_epoch} | Best MAE_nc: {best_cindex:.4f}")

    test(main_args, output_dir, model, train_loader)


if __name__ == "__main__":
    args = parse_args_cox()
    main(args)
