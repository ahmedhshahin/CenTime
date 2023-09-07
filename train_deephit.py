"""
DeepHit model training and evaluation.

Author: Ahmed H. Shahin
Date: 31/08/2023
"""

import os
from argparse import Namespace
from time import time
from typing import Any, Tuple

import numpy as np
import torch

from eval_utils import cindex, mae, rae
from losses import deephit_loss
from utils import (
    collect_batch_stats,
    get_data,
    get_grad_norm,
    parse_args,
    prepare_batch,
    prepare_epoch_stats,
    log_metrics,
    save_checkpoint,
    initialize,
    initialize_model_and_optimizer,
)

def parse_args_deephit() -> Namespace:
    """
    Get arguments from command line. Use the base parser in utils.py and add
      deephit-specific arguments, if any.

    Returns:
        args: Namespace object containing arguments.
    """
    parser = parse_args()
    parser.add_argument(
        "-ranking",
        default=0,
        type=int,
        help="Whether to use the ranking loss term in DeepHit.",
    )
    args = parser.parse_args()
    return args


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
    """
    model.train()
    device = train_args.device
    epoch_loss = 0.0
    all_preds, all_events, all_times = [], [], []
    for batch_idx, sample in enumerate(loader):
        img, clinical_data, time_to_event, event = prepare_batch(sample, device)

        optim.zero_grad()
        pred = model(img, clinical_data).softmax(dim=1)

        loss = deephit_loss(pred, event, time_to_event, ranking=train_args.ranking)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 100.0, error_if_nonfinite=True
        )
        optim.step()
        grad_norm = get_grad_norm(model)
        writer.add_scalar("grad_norm", grad_norm, batch_idx + epoch * len(loader))
        writer.add_scalar("loss_per_iteration", loss, batch_idx + epoch * len(loader))
        epoch_loss += loss * img.size(0)

        pred = pred.argmax(dim=1).reshape(-1) + 1  # convert from 0-indexed to 1-indexed
        collect_batch_stats(
            all_preds, all_events, all_times, pred, event, time_to_event
        )

    preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)
    # -preds because cindex function expects hazard scores, not survival scores
    cindex_score = cindex(times, events, -1 * preds)
    mae_nc = mae(preds, times, events, mode="uncens")
    return epoch_loss / len(loader.dataset), cindex_score, mae_nc


def validate_one_epoch(val_args, model, loader):
    """
    Validate the model for one epoch.

    Args:
        val_args: The command-line arguments or a configuration object.
        model: The PyTorch model to be validated.
        loader: The DataLoader for the validation set.

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

            pred = model(img, clinical_data).softmax(dim=1)
            loss = deephit_loss(pred, event, time_to_event, ranking=val_args.ranking)
            epoch_loss += loss * img.size(0)

            pred = pred.argmax(dim=1).reshape(-1) + 1
            collect_batch_stats(
                all_preds, all_events, all_times, pred, event, time_to_event
            )

        epoch_loss /= len(loader.dataset)
        preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)

        cindex_score = cindex(times, events, preds)
        mae_nc = mae(preds, times, events, mode="uncens")
        mae_c = mae(preds, times, events, mode="cens")
        rae_nc = rae(preds, times, events, mode="uncens")
        rae_c = rae(preds, times, events, mode="cens")
        return epoch_loss, cindex_score, rae_nc, rae_c, mae_nc, mae_c


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
        pid_sids: Patient IDs.
        dists: The softmax distribution over the classes.
    """
    model.eval()

    all_dists, all_events, all_times, all_pid_sids = [], [], [], []
    with torch.no_grad():
        for sample in loader:
            img, clinical_data, time_to_event, event = prepare_batch(sample, device)

            dist = model(img, clinical_data).softmax(dim=1).cpu().numpy()
            collect_batch_stats(
                all_dists, all_events, all_times, dist, event, time_to_event
            )
            all_pid_sids.extend(sample["pid_sid"])

        preds = np.argmax(all_dists, axis=1) + 1
        _, events, times = prepare_epoch_stats(None, all_events, all_times)
        pid_sids = np.array(all_pid_sids).reshape(-1)
        dists = np.array([dist.cpu().numpy() for dist in all_dists])
    return preds, events, times, pid_sids, dists


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
        mae_nc: The mean absolute error for uncensored data.
    """
    start_time =  time()
    (
        train_loss,
        cindex_train,
        mae_nc_train,
    ) = train_one_epoch(train_val_args, model, train_loader, optim, epoch, writer)

    (
        val_loss,
        cindex_val,
        rae_nc_val,
        rae_c_val,
        mae_nc_val,
        mae_c_val,
    ) = validate_one_epoch(train_val_args, model, val_loader)
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
        cindex_val,
        rae_nc_val,
        rae_c_val,
        mae_nc_val,
        mae_c_val,
    )
    save_checkpoint(
        model,
        optim,
        epoch,
        mae_nc_val,
        suffix="last",
        output_dir=output_dir,
        metric="mae_nc",
    )
    return mae_nc_val


def test(
    test_args: Any,
    output_dir: str,
    model: torch.nn.Module,
) -> None:
    """Test the model and print the results.

    Args:
        args: Command-line arguments.
        output_dir: Directory where the best model checkpoint is saved.
        model: The PyTorch model to be tested.
    """
    # Load the best model
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "checkpoint_best.pth"))["model"]
    )
    model.eval()

    # Get the test data loader
    test_loader = get_data(test_args, test=True)

    # Validate the model on the test set
    test_loss, cindex_score, rae_nc, rae_c, mae_nc, mae_c = validate_one_epoch(
        test_args, model, test_loader
    )

    # Log the test results
    print("Test results:")
    print(
        f"Test loss: {test_loss:.4f} | C-index: {cindex_score:.4f} | MAE_nc:\
              {mae_nc:.4f} | MAE_c: {mae_c:.4f} | RAE_nc: {rae_nc:.4f} | RAE_c:\
                  {rae_c:.4f}"
    )


def load_checkpoint(ckpt_path, device):
    """Load a checkpoint if provided and return relevant state variables."""

    # Initialize default values
    ckpt = None
    epoch = 0
    best_epoch = 0
    best_mae_nc = float("inf")  # Initialize to positive infinity

    # Try to load the checkpoint if provided
    if ckpt_path:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            print(f"Checkpoint loaded from {ckpt_path}")
        except FileNotFoundError:
            print(f"No checkpoint found at {ckpt_path}")

    return ckpt, epoch, best_epoch, best_mae_nc


def main(main_args):
    """
    Main function for training and testing the model.
    """
    output_dir, writer = initialize(main_args)

    main_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {main_args.device}")

    # load checkpoint, if provided
    ckpt, epoch, best_epoch, best_mae_nc = load_checkpoint(
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

        mae_nc = train_and_validate_epoch(
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

        if mae_nc < best_mae_nc:
            print("Saving best model...")
            best_mae_nc = mae_nc
            best_epoch = epoch
            save_checkpoint(
                model,
                optim,
                epoch,
                best_mae_nc,
                suffix="best",
                output_dir=output_dir,
                metric="mae_nc",
            )
        print()
    print(f"Best epoch: {best_epoch} | Best MAE_nc: {best_mae_nc:.4f}")

    test(main_args, output_dir, model)


if __name__ == "__main__":
    deephit_args = parse_args_deephit()
    main(deephit_args)
