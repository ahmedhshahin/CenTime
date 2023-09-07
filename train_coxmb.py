"""
CoxMB (Cox with memory banks) model training and evaluation.

Author: Ahmed H. Shahin
Date: 31/08/2023
"""

from argparse import ArgumentParser
from collections import deque
from time import time

import numpy as np
import torch

from eval_utils import cindex, get_median_survival_cox, mae, rae
from likelihoods import cox_loss
from train_cox import load_checkpoint, test
from utils import (
    collect_batch_stats,
    get_data,
    get_grad_norm,
    initialize,
    initialize_model_and_optimizer,
    log_metrics,
    parse_args,
    prepare_batch,
    prepare_epoch_stats,
    save_checkpoint,
)


def parse_args_coxmb() -> ArgumentParser:
    """
    Get arguments from command line.

    Returns:
        args: Namespace object containing arguments.
    """
    parser = parse_args()
    coxmb_args = parser.add_argument_group("CoxMB arguments")
    coxmb_args.add_argument(
        "-K",
        default=1.0,
        type=float,
        help="proportion of training data to be stored in memory bank",
    )
    args = parser.parse_args()
    return args


def update_memory_bank(memory_bank, img, time_to_event, event, risk):
    """
    Update the memory bank.
    """
    for i in range(len(img)):
        memory_bank[0].append(risk[i])
        memory_bank[1].append(event[i])
        memory_bank[2].append(time_to_event[i])


def compute_loss(memory_bank):
    """
    Compute the Cox loss on the memory bank entries.
    """
    loss = cox_loss(
        torch.cat(list(memory_bank[0])),
        torch.cat(list(memory_bank[1])),
        torch.cat(list(memory_bank[2])),
    )
    return loss


def train_one_epoch(train_args, model, loader, optim, epoch, writer, memory_bank=None):
    """
    Train the model for one epoch.

    Args:
        train_args: The command-line arguments or a configuration object.
        model: The PyTorch model to be trained.
        loader: The DataLoader for the training set.
        optim: The optimizer.
        epoch: The current epoch number.
        writer: The TensorBoard writer object for logging.
        memory_bank: The memory bank for the model.

    Returns:
        epoch_loss: The average loss for this epoch.
        cindex: The concordance index for the training set.
        mae_nc: The mean absolute error for uncensored data.
        preds: The model's predictions.
        events: The event occurrences.
        times: The time-to-event data.
        memory_bank: The updated memory bank.
    """
    model.train()
    device = train_args.device
    epoch_loss = 0.0
    all_preds, all_events, all_times = [], [], []
    for batch_idx, sample in enumerate(loader):
        img, clinical_data, time_to_event, event = prepare_batch(sample, device)
        optim.zero_grad()
        risk = model(img, clinical_data)

        update_memory_bank(memory_bank, img, time_to_event, event, risk)

        loss = compute_loss(memory_bank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 100.0, error_if_nonfinite=True
        )
        grad_norm = get_grad_norm(model)
        optim.step()

        # log the training loss and gradient norm
        writer.add_scalar("grad_norm", grad_norm, batch_idx + epoch * len(loader))
        writer.add_scalar("loss_per_iteration", loss, batch_idx + epoch * len(loader))

        # accumulate loss
        epoch_loss += loss.item() * event.sum().item()

        collect_batch_stats(
            all_preds, all_events, all_times, risk, event, time_to_event
        )

        # release gradients in the memory bank, to save memory
        for i in range(len(img)):
            memory_bank[0][i - len(img)] = risk[i].detach()

    preds, events, times = prepare_epoch_stats(all_preds, all_events, all_times)
    cindex_score = cindex(times, events, preds)
    preds_times = get_median_survival_cox(preds, events, times, preds)
    mae_nc = mae(preds_times, times, events, "uncens")
    return (
        epoch_loss / loader.dataset.event.sum(),
        cindex_score,
        mae_nc,
        preds,
        events,
        times,
        memory_bank,
    )


def validate_one_epoch(
    val_args, model, loader, tr_preds, tr_events, tr_times, memory_bank=None
):
    """
    Validate the model for one epoch.

    Args:
        val_args: The command-line arguments or a configuration object.
        model: The PyTorch model to be validated.
        loader: The DataLoader for the validation set.
        tr_preds: The training set predictions.
        tr_events: The training set event occurrences.
        tr_times: The training set time-to-event data.
        memory_bank: The validation memory bank for the model.

    Returns:
        epoch_loss: The average loss for this epoch.
        cindex: The concordance index for the validation set.
        rae_nc: The relative absolute error for uncensored data.
        rae_c: The relative absolute error for censored data.
        mae_nc: The mean absolute error for uncensored data.
        mae_c: The mean absolute error for censored data.
        memory_bank: The updated memory bank.
    """
    model.eval()
    device = val_args.device
    epoch_loss = 0.0
    all_preds, all_events, all_times = [], [], []
    with torch.no_grad():
        for sample in loader:
            img, clinical_data, time_to_event, event = prepare_batch(sample, device)

            risk = model(img, clinical_data)

            # update memory bank
            if memory_bank is not None:
                update_memory_bank(memory_bank, img, time_to_event, event, risk)

                loss = compute_loss(memory_bank)
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
        rae_nc = rae(preds_times, times, events, "uncens")
        rae_c = rae(preds_times, times, events, "cens")
        mae_c = mae(preds_times, times, events, "cens")
        return epoch_loss, cindex_score, rae_nc, rae_c, mae_nc, mae_c, memory_bank


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
    train_memory_bank,
    val_memory_bank,
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
        train_memory_bank: The training memory bank for the model.
        val_memory_bank: The validation memory bank for the model.

    Returns:
        cindex: The concordance index for the validation set.
    """
    start_time =  time()
    (
        train_loss,
        cindex_train,
        mae_nc_train,
        train_preds,
        train_events,
        train_times,
        train_memory_bank,
    ) = train_one_epoch(
        train_val_args, model, train_loader, optim, epoch, writer, train_memory_bank
    )

    (
        val_loss,
        cindex_score,
        rae_nc,
        rae_c,
        mae_nc,
        mae_c,
        val_memory_bank,
    ) = validate_one_epoch(
        train_val_args,
        model,
        val_loader,
        train_preds,
        train_events,
        train_times,
        val_memory_bank,
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
    save_checkpoint(model, optim, epoch, cindex_score, suffix="last",
                     output_dir=output_dir, metric="cindex")
    return cindex_score


def initialize_memory_bank(main_args, train_loader):
    """
    Initialize the memory bank.
    """
    assert main_args.K <= 1.0
    main_args.K = int(main_args.K * len(train_loader.dataset))
    train_memory_pred = deque(maxlen=main_args.K)
    train_memory_times = deque(maxlen=main_args.K)
    train_memory_events = deque(maxlen=main_args.K)
    val_memory_pred = deque(maxlen=main_args.K)
    val_memory_times = deque(maxlen=main_args.K)
    val_memory_events = deque(maxlen=main_args.K)
    train_memory_bank = (train_memory_pred, train_memory_events, train_memory_times)
    val_memory_bank = (val_memory_pred, val_memory_events, val_memory_times)
    return train_memory_bank, val_memory_bank


def main(main_args):
    """
    Main function.
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

    # initialize the memory banks
    train_memory_bank, val_memory_bank = initialize_memory_bank(main_args, train_loader)

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
            train_memory_bank,
            val_memory_bank,
        )
        if cindex_score > best_cindex:
            print("Saving best model...")
            best_cindex = cindex_score
            best_epoch = epoch
            save_checkpoint(
                model, optim, epoch, best_cindex, suffix="best", output_dir=output_dir,
                  metric="cindex"
            )
        print()
    print(f"Best epoch: {best_epoch} | Best MAE_nc: {best_cindex:.4f}")

    test(main_args, output_dir, model, train_loader)


if __name__ == "__main__":
    args_coxmb = parse_args_coxmb()
    main(args_coxmb)
