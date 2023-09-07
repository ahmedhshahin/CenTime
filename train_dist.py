"""
Training and evaluation using distributional survival analysis, CenTime and the
 classical losses.

Author: Ahmed H. Shahin
Date: 31/08/2023
"""

import os
from argparse import Namespace
from time import time
from typing import Any

import numpy as np
import torch

from eval_utils import cindex, mae, rae
from losses import (
    classical_loss,
    centime_loss,
    get_mean_prediction,
    get_probs,
)
from utils import (
    collect_batch_stats,
    detect_nan,
    get_data,
    get_grad_norm,
    initialize,
    initialize_model_and_optimizer,
    parse_args,
    prepare_batch,
    prepare_epoch_stats,
    save_checkpoint,
)


def parse_args_dist() -> Namespace:
    """
    Get arguments from command line. Use the base parser in utils.py and add specific
      arguments for this script.

    Returns:
        args: Namespace object containing arguments.
    """
    parser = parse_args()
    parser.add_argument(
        "-loss",
        type=str,
        default="centime",
        help="Loss function to use. Options: centime, classical",
    )
    parser.add_argument(
        "-var", type=float, default=144, help="Fixed variance of the distribution"
    )
    args = parser.parse_args()
    return args


def get_loss(loss_fn: str):
    """
    Get the loss function.

    Args:
        loss_fn (str): loss function to use. Options: classical, centime
    """
    if loss_fn == "classical":
        return classical_loss
    if loss_fn == "centime":
        return centime_loss
    raise ValueError(f"Invalid loss function: {loss_fn}")


def train_one_epoch(train_args, model, loader, optim, loss, epoch, writer):
    """
    Train the model for one epoch.

    Args:
        train_args: The command-line arguments or a configuration object.
        model: The PyTorch model to be trained.
        loader: The DataLoader for the training set.
        optim: The optimizer.
        loss: The loss function.
        epoch: The current epoch number.
        writer: The TensorBoard writer object for logging.

    Returns:
        epoch_loss: The total average loss for this epoch.
        epoch_loss_cens: The average loss for censored samples.
        epoch_loss_uncens: The average loss for uncesnored samples.
        cindex: The concordance index for the training set.
        mae_nc: The mean absolute error for uncensored data.
    """
    model.train()
    device = train_args.device
    epoch_loss, epoch_loss_cens, epoch_loss_uncens = 0.0, 0.0, 0.0
    all_preds, all_events, all_times = [], [], []
    for batch_idx, sample in enumerate(loader):
        img, clinical_data, time_to_event, event = prepare_batch(sample, device)

        optim.zero_grad()
        loc = model(img, clinical_data)
        var = torch.ones_like(loc) * train_args.var

        # detect inf or nan predictions
        detect_nan(loc)

        loss_cens, loss_uncens = loss(
            loc, var, event, time_to_event, train_args.tmax, train_args.distribution
        )
        loss_val = loss_uncens + loss_cens
        # detect inf loss
        detect_nan(loss_val)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optim.step()
        grad_norm = get_grad_norm(model)

        writer.add_scalar("grad_norm", grad_norm, batch_idx + epoch * len(loader))
        writer.add_scalar(
            "loss_per_iteration", loss_val, batch_idx + epoch * len(loader)
        )
        epoch_loss += (loss_cens + loss_uncens) * img.size(0)
        epoch_loss_cens += loss_cens * img.size(0)
        epoch_loss_uncens += loss_uncens * img.size(0)

        dists = get_probs(loc, var, train_args.tmax, train_args.distribution)
        pred = get_mean_prediction(dists, train_args.tmax)
        collect_batch_stats(
            all_preds, all_events, all_times, pred, event, time_to_event
        )

    preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)
    cindex_score = cindex(times, events, -1 * preds)
    mae_nc = mae(preds, times, events, mode="uncens")

    epoch_loss /= len(loader.dataset)
    epoch_loss_cens /= len(loader.dataset) - loader.dataset.event.sum()
    epoch_loss_uncens /= loader.dataset.event.sum()
    return epoch_loss, epoch_loss_cens, epoch_loss_uncens, cindex_score, mae_nc


def validate_one_epoch(val_args, model, loader, loss):
    """
    Validate the model for one epoch.

    Args:
        val_args: The command-line arguments or a configuration object.
        model: The PyTorch model to be validated.
        loader: The DataLoader for the validation set.
        loss: The loss function.

    Returns:
        epoch_loss: The total average loss for this epoch.
        epoch_loss_cens: The average loss for censored samples.
        epoch_loss_uncens: The average loss for uncesnored samples.
        cindex: The concordance index for the training set.
        rae_nc: The relative absolute error for uncensored data.
        rae_c: The relative absolute error for censored data.
        mae_nc: The mean absolute error for uncensored data.
        mae_c: The mean absolute error for censored data.
    """
    model.eval()
    device = val_args.device
    epoch_loss, epoch_loss_cens, epoch_loss_uncens = 0.0, 0.0, 0.0
    all_preds, all_events, all_times = [], [], []
    with torch.no_grad():
        for sample in loader:
            img, clinical_data, time_to_event, event = prepare_batch(sample, device)
            loc = model(img, clinical_data)
            var = torch.ones_like(loc) * val_args.var

            # detect inf or nan predictions
            detect_nan(loc)

            loss_cens, loss_uncens = loss(
                loc, var, event, time_to_event, val_args.tmax, val_args.distribution
            )
            loss_val = loss_uncens + loss_cens

            # detect inf or nan loss
            detect_nan(loss_val)

            epoch_loss += (loss_cens + loss_uncens) * img.size(0)
            epoch_loss_cens += loss_cens * img.size(0)
            epoch_loss_uncens += loss_uncens * img.size(0)

            probs = get_probs(loc, var, val_args.tmax, val_args.distribution)
            pred = get_mean_prediction(probs, val_args.tmax)
            collect_batch_stats(
                all_preds, all_events, all_times, pred, event, time_to_event
            )

        preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)

        epoch_loss /= len(loader.dataset)
        epoch_loss_cens /= len(loader.dataset) - loader.dataset.event.sum()
        epoch_loss_uncens /= loader.dataset.event.sum()

        cindex_score = cindex(times, events, -1 * preds)
        rae_nc = rae(preds, times, events, mode="uncens")
        rae_c = rae(preds, times, events, mode="cens")
        mae_nc = mae(preds, times, events, mode="uncens")
        mae_c = mae(preds, times, events, mode="cens")
        return (
            epoch_loss,
            epoch_loss_cens,
            epoch_loss_uncens,
            cindex_score,
            rae_nc,
            rae_c,
            mae_nc,
            mae_c,
        )


def load_checkpoint(ckpt_path: str, device: Any):
    """Load a checkpoint if provided and return relevant state variables."""

    # Initialize default values
    ckpt = None
    epoch = 0
    best_epoch = 0
    best_mae_nc = float("inf")

    # Try to load the checkpoint if provided
    if ckpt_path:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            print(f"Checkpoint loaded from {ckpt_path}")
        except FileNotFoundError:
            print(f"No checkpoint found at {ckpt_path}")

    return ckpt, epoch, best_epoch, best_mae_nc


def get_preds(
    preds_args: Namespace, model: torch.nn.Module, loader: torch.utils.data.DataLoader
):
    """Get model predictions, events, and times for a given data loader.

    Args:
        preds_args: The command-line arguments or a configuration object.
        model: The PyTorch model to be evaluated.
        loader: DataLoader for the dataset.

    Returns:
        preds: Predictions from the model.
        events: Event occurrences.
        times: Time to events.
        pid_sids: Patient IDs.
        locs: The predicted location parameters.
    """
    model.eval()
    device = preds_args.device
    all_preds, all_events, all_times, all_pid_sids, all_locs = [], [], [], [], []

    with torch.no_grad():
        for sample in loader:
            img, clinical_data, time_to_event, event = prepare_batch(sample, device)

            loc = model(img, clinical_data)
            var = torch.ones_like(loc) * preds_args.var

            all_locs.extend(loc)
            dists = get_probs(loc, var, preds_args.tmax, preds_args.distribution)
            pred = get_mean_prediction(dists, preds_args.tmax)
            all_pid_sids.extend(sample["pid_sid"])

            collect_batch_stats(
                all_preds, all_events, all_times, pred, event, time_to_event
            )

        preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)
        pid_sids = np.array(all_pid_sids).reshape(-1)
        locs = np.array([loc.cpu().numpy() for loc in all_locs]).reshape(-1)
    return preds, events, times, pid_sids, locs


def log_metrics(
    epochs,
    writer,
    epoch,
    start_time,
    train_loss,
    train_loss_cens,
    train_loss_uncens,
    cindex_train,
    mae_nc_train,
    val_loss,
    val_loss_cens,
    val_loss_uncens,
    cindex_score,
    rae_nc,
    rae_c,
    mae_nc,
    mae_c,
):
    """Log metrics to TensorBoard and print them to the standard output.

    Args:
        args: Command-line arguments or equivalent.
        writer: TensorBoard writer object.
        epoch: Current epoch number.
        t1: Start time for the epoch.
        train_loss: Training loss.
        train_loss_cens: Training loss for censored samples.
        train_loss_uncens: Training loss for uncensored samples.
        cindex_train: C-index for the training set.
        mae_nc_train: MAE for the training set (uncensored).
        val_loss: Validation loss.
        val_loss_cens: Validation loss for censored samples.
        val_loss_uncens: Validation loss for uncensored samples.
        cindex_score: C-index for the validation set.
        rae_nc: RAE for the validation set (uncensored).
        rae_c: RAE for the validation set (censored).
        mae_nc: MAE for the validation set (uncensored).
        mae_c: MAE for the validation set (censored).
    """
    print(
        f"""
        Epoch {epoch}/{epochs} (time: {(time() - start_time) / 60:.2f}m):
        train_loss: {train_loss:.4f}, train_loss_cens: {train_loss_cens:.4f},\
              train_loss_uncens: {train_loss_uncens:.4f}
        val_loss: {val_loss:.4f}, val_loss_cens: {val_loss_cens:.4f},\
              val_loss_uncens: {val_loss_uncens:.4f}
        cindex: {cindex_score:.4f}, rae_nc: {rae_nc:.4f}, rae_c: {rae_c:.4f},\
              mae_nc: {mae_nc:.4f}, mae_c: {mae_c:.4f}"""
    )

    writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
    writer.add_scalars(
        "loss_cens", {"train": train_loss_cens, "val": val_loss_cens}, epoch
    )
    writer.add_scalars(
        "loss_uncens", {"train": train_loss_uncens, "val": val_loss_uncens}, epoch
    )
    writer.add_scalars("cindex", {"train": cindex_train, "val": cindex_score}, epoch)
    writer.add_scalars("mae_nc", {"train": mae_nc_train, "val": mae_nc}, epoch)
    writer.add_scalar("mae_c", mae_c, epoch)


def train_and_validate_epoch(
    train_val_args,
    output_dir,
    writer,
    epoch,
    model,
    optim,
    sched,
    loss,
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
        loss: The loss function.
        train_loader: The DataLoader for the training set.
        val_loader: The DataLoader for the validation set.

    Returns:
        mae_nc: The mean absolute error for uncensored data.
    """
    start_time =  time()
    (
        train_loss,
        train_loss_cens,
        train_loss_uncens,
        cindex_train,
        mae_nc_train,
    ) = train_one_epoch(train_val_args, model, train_loader, optim, loss, epoch, writer)

    (
        val_loss,
        val_loss_cens,
        val_loss_uncens,
        cindex_val,
        rae_nc_val,
        rae_c_val,
        mae_nc_val,
        mae_c_val,
    ) = validate_one_epoch(train_val_args, model, val_loader, loss)
    sched.step()

    log_metrics(
        train_val_args.epochs,
        writer,
        epoch,
        start_time,
        train_loss,
        train_loss_cens,
        train_loss_uncens,
        cindex_train,
        mae_nc_train,
        val_loss,
        val_loss_cens,
        val_loss_uncens,
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
    loss: Any,
) -> None:
    """
    Test the model and print the results.

    Args:
        args: Command-line arguments.
        output_dir: Directory where the best model checkpoint is saved.
        model: The PyTorch model to be tested.
        loss: The loss function.
    """
    # Load the best model
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "checkpoint_best.pth"))["model"]
    )
    model.eval()

    # Get the test data loader
    test_loader = get_data(test_args, test=True)

    # Validate the model on the test set
    test_loss, _, _, cindex_score, rae_nc, rae_c, mae_nc, mae_c = validate_one_epoch(
        test_args, model, test_loader, loss
    )

    # Log the test results
    print("Test results:")
    print(
        f"Test loss: {test_loss:.4f} | C-index: {cindex_score:.4f} | MAE_nc:\
              {mae_nc:.4f} | MAE_c: {mae_c:.4f} | RAE_nc: {rae_nc:.4f} | RAE_c:\
                  {rae_c:.4f}"
    )


def main(main_args):
    """
    Main function for training and testing the model.
    """
    output_dir, writer = initialize(main_args)

    main_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {main_args.device}")

    # load checkpoint, if provided
    ckpt, epoch, best_epoch, best_mae_nc = load_checkpoint(
        main_args.ckpt, main_args.device
    )

    # model and optimizer
    model, optim, sched = initialize_model_and_optimizer(main_args, ckpt)

    # load the data
    train_loader, val_loader = get_data(main_args)

    # get the loss function
    loss = get_loss(main_args)

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
            loss,
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

    test(main_args, output_dir, model, loss)


if __name__ == "__main__":
    dist_args = parse_args_dist()
    main(dist_args)
