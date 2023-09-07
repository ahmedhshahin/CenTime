"""
Utility functions.

Author: Ahmed H. Shahin
Date: 31/8/2023
"""

import datetime
import json
import os
import random
from argparse import ArgumentParser, Namespace
from glob import glob
from time import time
from typing import Optional

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from scipy import special
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dataset
from imputation.train_imputation_model import condexp, predict_missing_values
from model import Model


def detect_nan(input_tensor):
    """
    Detect inf or nan. For debugging purposes.

    Args:
        x: Tensor
    """
    if torch.isinf(input_tensor).any() or torch.isnan(input_tensor).any():
        raise ValueError("inf or nan detected")


def prepare_epoch_stats(all_preds, all_events, all_times):
    """
    Prepare epoch statistics to be used for computing metrics (e.g. C-Index). Convert
      lists to numpy arrays.

    Args:
        all_preds: List of model predictions.
        all_events: List of event indicators.
        all_times: List of time-to-event or time-to-censoring.

    Returns:
        preds: Numpy array of model predictions.
        events: Numpy array of event indicators.
        times: Numpy array of time-to-event or time-to-censoring.
    """
    preds = (
        np.array([pred.data.cpu().numpy() for pred in all_preds]).reshape(-1)
        if all_preds is not None
        else None
    )
    events = (
        np.array([event.cpu().numpy() for event in all_events]).reshape(-1).astype(bool)
        if all_events is not None
        else None
    )
    times = (
        np.array([time.cpu().numpy() for time in all_times]).reshape(-1)
        if all_times is not None
        else None
    )
    return preds, events, times


def collect_batch_stats(all_preds, all_events, all_times, pred, event, time_to_event):
    """
    Collect batch statistics to be used for computing metrics (e.g. C-Index).

    Args:
        all_preds: List of model predictions.
        all_events: List of event indicators.
        all_times: List of time-to-event or time-to-censoring.
    """
    all_preds.extend(pred)
    all_events.extend(event)
    all_times.extend(time_to_event)


def prepare_batch(sample, device):
    """
    Prepare a batch of data for training, moved to the specified device.

    Args:
        sample: Batch of data (dict).

    Returns:
        img: Image tensor.
        clinical_data: Clinical data tensor.
        time_to_event: Time-to-event tensor.
        event: Event indicator tensor.
    """
    img, clinical_data, time_to_event, event = (
        sample["img"],
        sample["clinical_data"],
        sample["time_to_event"].type(torch.int64),
        sample["event"],
    )
    img, clinical_data, time_to_event, event = (
        img.to(device, non_blocking=True),
        clinical_data.to(device, non_blocking=True),
        time_to_event.to(device, non_blocking=True),
        event.to(device, non_blocking=True),
    )
    return img, clinical_data, time_to_event, event


def initialize_model_and_optimizer(model_args, ckpt):
    """Initialize the model and optimizer.

    Args:
        model_args: Model arguments.
        ckpt: Checkpoint to load.
    """
    # Initialize the model
    model = get_model(model_args, 1, ckpt)

    # Initialize the optimizer
    optim = get_optimizer(model_args, model, checkpoint=None)  # TODO: hacky
    sched = get_scheduler(model_args, optim)

    return model, optim, sched


def log_metrics(
    epochs,
    writer,
    epoch,
    start_time,
    train_loss,
    cindex_train,
    mae_nc_train,
    val_loss,
    cindex,
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
        start_time: Start time for the epoch.
        train_loss: Training loss.
        cindex_train: C-index for the training set.
        mae_nc_train: MAE for the training set (uncensored).
        val_loss: Validation loss.
        cindex: C-index for the validation set.
        rae_nc: RAE for the validation set (uncensored).
        rae_c: RAE for the validation set (censored).
        mae_nc: MAE for the validation set (uncensored).
        mae_c: MAE for the validation set (censored).
    """
    print(
        f"""
        Epoch {epoch}/{epochs} (time: {(time() - start_time) / 60:.2f}m):
        train_loss: {train_loss:.4f}
        val_loss: {val_loss:.4f}
        cindex: {cindex:.4f}, rae_nc: {rae_nc:.4f}, rae_c: {rae_c:.4f},\
              mae_nc: {mae_nc:.4f}, mae_c: {mae_c:.4f}"""
    )

    writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
    writer.add_scalars("cindex", {"train": cindex_train, "val": cindex}, epoch)
    writer.add_scalars("mae_nc", {"train": mae_nc_train, "val": mae_nc}, epoch)
    writer.add_scalar("mae_c", mae_c, epoch)


def initialize(init_args):
    """Initialize the training environment.

    Args:
        init_args: Arguments for initialization.
    """

    # Fix random seed for reproducibility
    fix_seed(123)

    # Create the output directory
    init_args.exp_name = init_args.exp_name + "_" if init_args.exp_name != "" else ""
    output_dir = os.path.join(
        init_args.output_dir,
        init_args.exp_name
        + init_args.model_name
        + "_"
        + ("seg" if init_args.segment else "noseg")
        + "_"
        + ("fold" + str(init_args.fold))
        + "_"
        + str(init_args.lr)
        + "_"
        + ("CLINICAL_" if init_args.clinical_data else "")
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Handle clinical data options. If clinical data is used,
    # find the imaging model checkpoint path.
    if init_args.clinical_data:
        init_args.n_clinical_data = 20 if init_args.missing_data_indicator else 10
        exps = glob(init_args.output_dir + "/*")
        imaging_model_path = [
            e
            for e in exps
            if os.path.split(e)[1].startswith(
                os.path.split(output_dir)[1][
                    : os.path.split(output_dir)[1].find("fold") + 5
                ]
            )
        ]
        imaging_model_path = [e for e in imaging_model_path if "CLINICAL" not in e]

        if len(imaging_model_path) != 1:
            raise ValueError(
                f"Expected exactly one imaging model path, found: {len(imaging_model_path)}"
            )
        init_args.checkpoint_path = os.path.join(
            imaging_model_path[0], "checkpoint_best.pth"
        )
    else:
        init_args.n_clinical_data = 0

    # Initialize TensorBoard writer
    writer = SummaryWriter(output_dir)

    # Save and print arguments
    with open(os.path.join(output_dir, "args.json"), "w") as file:
        json.dump(vars(init_args), file, indent=4)
    print(f"Arguments: {vars(init_args)}")

    return output_dir, writer


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    suffix: str = "last",
    output_dir: Optional[str] = None,
    metric: Optional[str] = "cindex",
):
    """Save a model checkpoint.

    Args:
        model: PyTorch model to be saved.
        optimizer: PyTorch optimizer to be saved.
        epoch: Current epoch number.
        best_metric: Best metric value.
        suffix: Suffix to append to the checkpoint filename.
        output_dir: Directory where the checkpoint will be saved. If None,
          the current directory is used.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_" + metric: best_metric,
    }
    if output_dir is None:
        output_dir = "."
    torch.save(state, os.path.join(output_dir, f"checkpoint_{suffix}.pth"))
    print(f"Checkpoint saved to {os.path.join(output_dir, f'checkpoint_{suffix}.pth')}")


def parse_args() -> Namespace:
    """
    Get arguments from command line.

    Returns:
        parser: parser object
    """
    parser = ArgumentParser()
    data_args = parser.add_argument_group("Data Options")
    data_args.add_argument("-data_path", type=str, help="path to the dataset folder")
    data_args.add_argument("-segment", type=int, default=0, help="segmentation or not")
    data_args.add_argument("-fold", type=int, default=0, help="fold number")
    data_args.add_argument(
        "-n", type=int, default=-1, help="number of samples to use. -1 for all"
    )
    data_args.add_argument(
        "-p_aug", default=0.0, type=float, help="probability of augmentation"
    )
    data_args.add_argument(
        "-hist_match", default=0, type=int, help="histogram matching"
    )
    data_args.add_argument(
        "-max_res", default=3, type=int, help="maximum slice thickness"
    )
    data_args.add_argument(
        "-p_uncens",
        default=1.0,
        type=float,
        help="propotion of uncensored samples to use, default 1.0 (all)",
    )
    data_args.add_argument(
        "-clinical_data", default=0, type=int, help="use clinical data or not"
    )
    data_args.add_argument(
        "-missing_data_indicator",
        default=0,
        type=int,
        help="use missing data indicator or not",
    )

    training_args = parser.add_argument_group("Training Options")
    training_args.add_argument("-batch_size", type=int, default=16, help="batch size")
    training_args.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    training_args.add_argument("-tmax", type=int, default=156, help="tmax")
    training_args.add_argument("-epochs", type=int, default=100, help="epochs")
    training_args.add_argument(
        "-patience",
        type=int,
        default=50,
        help="Epochs to wait before halting training when validation performance has not improved",
    )
    training_args.add_argument(
        "-min_epochs", type=int, default=200, help="minimum number of epochs"
    )
    training_args.add_argument("-wd", default=5e-4, type=float, help="weight decay")

    model_args = parser.add_argument_group("Model Options")
    model_args.add_argument(
        "-n_base_filters", type=int, default=16, help="number of base filters"
    )
    model_args.add_argument(
        "-act", default="relu", type=str, help="activation function"
    )
    model_args.add_argument(
        "-n_layers", default=3, type=int, help="number of residual blocks"
    )
    model_args.add_argument(
        "-in_filters", default=1, type=int, help="number of input filters"
    )
    model_args.add_argument(
        "-out_filters",
        default=128,
        type=int,
        help="number of output filters before the fully-connected",
    )

    misc_args = parser.add_argument_group("Miscellaneous Options")
    misc_args.add_argument(
        "-output_dir", type=str, default="exps", help="output directory"
    )
    misc_args.add_argument("-exp_name", type=str, default="")
    misc_args.add_argument(
        "-checkpoint_path", type=str, default="", help="checkpoint to load"
    )

    return parser


def create_dataset(
    dataset_args, split, clinical_normalizer=None, clinical_encoder=None, load_imgs=True
):
    """
    Creates the dataset class.

    Args:
        dataset_args: Dataset arguments.
        split: Dataset split ('train', 'val', or 'test').
        clinical_normalizer: Clinical data normalizer.
        clinical_encoder: Clinical data encoder.
        load_imgs: Whether to load images or not.

    Returns:
        Dataset class.
    """
    return dataset.OSICDataset(
        dataset_args.data_path,
        split,
        segment=dataset_args.segment,
        fold=dataset_args.fold,
        num_patients=dataset_args.n,
        p_aug=vars(dataset_args).get("p_aug", 0.0),
        hist_match=vars(dataset_args).get("hist_match", 0),
        max_res=vars(dataset_args).get("max_res", 3),
        p_uncens=vars(dataset_args).get("p_uncens", 1.0),
        clinical_normalizer=clinical_normalizer,
        clinical_encoder=clinical_encoder,
        load_imgs=load_imgs,
        use_missing_indicator=dataset_args.missing_data_indicator,
    )


def get_data(args, test=False, val=True):
    """
    Get the data loaders.

    Args:
        args: Arguments.
        test: Whether to get the test loader or not. If True, only the test loader is
          returned.
        val: Whether to get the validation loader or not. If True, the training and
          validation loaders are returned. If False, only the training loader is
            returned.
    """
    if not test:
        train_dataset = create_dataset(args, "train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        if val:
            val_dataset = create_dataset(
                args,
                "val",
                clinical_normalizer=train_dataset.clinical_normalizer,
                clinical_encoder=train_dataset.clinical_encoder,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            return train_loader, val_loader

        return train_loader

    train_dataset_for_test = create_dataset(args, "train", load_imgs=False)
    test_dataset = create_dataset(
        args,
        "test",
        clinical_normalizer=train_dataset_for_test.clinical_normalizer,
        clinical_encoder=train_dataset_for_test.clinical_encoder,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return test_loader


def get_optimizer(args, model, checkpoint=None):
    """
    Get the optimizer.

    Args:
        Args: Arguments.
        model: Model.
        checkpoint: Checkpoint to load.

    Returns:
        optim: Optimizer.
    """
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if checkpoint is not None:
        optim.load_state_dict(checkpoint["optimizer"])
    return optim


def get_scheduler(args, optim):
    """
    Get the LR scheduler.

    Args:
        args: Arguments.
        optim: Optimizer.

    Returns:
        scheduler: LR scheduler.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, args.epochs, verbose=True
    )
    return scheduler


def get_grad_norm(model):
    """
    Get the L2 norm of the gradients, for debugging purposes.

    Args:
        model: Model.

    Returns:
        grad_norm: L2 norm of the gradients.
    """
    total_norm = 0
    for par in model.parameters():
        if par.grad is not None:
            param_norm = par.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def fix_seed(seed):
    """
    Fix the random seed for reproducibility.

    Args:
        seed: Seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def load_args_from_json(json_path):
    """
    Load arguments from a json file.

    Args:
        json_path: Path to the json file.

    Returns:
        args: Arguments.
    """
    with open(json_path, "r") as file:
        args = json.load(file)
    return Namespace(**args)


def histogram_matching_sitk(
    mov_scan, ref_scan, histogram_levels=2048, match_points=100, set_th_mean=True
):
    """
    Apply histogram matching following the method in Nyul et al 2001 (ITK implementation)

    Args:
        mov_scan: np.array containing the image to normalize
        ref_scan np.array containing the reference image
        histogram levels: number of histogram levels
        match_points: number of matched points
        set_th_mean: whether to set the threshold at the mean intensity of the reference image

    Returns:
        histogram matched image
    """

    # convert np arrays into itk image objects
    ref = sitk.GetImageFromArray(ref_scan.astype("float32"))
    mov = sitk.GetImageFromArray(mov_scan.astype("float32"))

    # perform histogram matching
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(ref.GetPixelID())

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(histogram_levels)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.SetThresholdAtMeanIntensity(set_th_mean)
    matched_vol = matcher.Execute(mov, ref)

    return sitk.GetArrayFromImage(matched_vol)


def load_scan(path):
    """
    Load a scan from an h5 file.

    Args:
        path: Path to the h5 file.

    Returns:
        scan: Scan.
    """
    file = h5py.File(path, "r")
    scan = np.array(file.get("img"))
    return scan


def normalize_scan(img, min_hu=-1024, max_hu=350):
    """
    Normalize and apply HU windowing to a scan.

    Args:
        img: Scan.
        min_hu: Minimum HU value.
        max_hu: Maximum HU value.

    Returns:
        img: Normalized scan.
    """
    img = np.clip(img, min_hu, max_hu)
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255.0
    return img.astype(np.uint8)


def compute_log_prob(logp_h, logp_xgh, record, num_categories):
    """
    Compute log p(x_m|x_o) by summing over the observed features only. The missing
      features are added in the next step.

    Args:
        logp_h: the distribution of the latent variable h (np.array of shape (H,)).
        logp_xgh: the distribution of x given h. A list of length D, each element is a
          matrix of shape (H, num_categories[d]), where num_categories[d] is the number
            of possible categories of feature d.
        record: a record of x (np.array of shape (D,)).
        num_categories: the number of possible categories of each feature (np.array of
          shape (D,)).

    Returns:
        logpxm_given_xo: log p(x_m|x_o) (np.array of shape (H,)).
    """
    logp_xm_given_xo = logp_h.copy()
    for category in range(num_categories.max()):
        class_indices = np.where(record == category)[0]
        for f_idx in class_indices:
            logp_xm_given_xo += logp_xgh[f_idx][:, category]
    return logp_xm_given_xo


def impute_missing_values(arr, p_h, p_xgh, num_categories, selection="max", eps=1e-10):
    """
    Impute missing values in x using the trained latent-variable model.
    p(x_m|x_o) \\probto \\sum_h p(x_m,x_o|h)p(h) = \\sum_h p(x_m|h)p(x_o|h)p(h)
    log p(x_m|x_o) = logsumexp_h (log p(x_m|h) + log p(x_o|h) + log p(h))

    Args:
        arr: **discrete** data matrix with missing values (np.nan) of shape (N,D). Can
          be pre-discritized using the discretizer object used in training.
        p_h: the distribution of the latent variable h (np.array of shape (H,))
        p_xgh: the distribution of x given h. A list of length D, each element is a
          matrix of shape (H, num_categories[d]), where num_categories[d] is the number
            of possible categories of feature d.
        num_categories: the number of possible categories of each feature (np.array of
          shape (D,)).
        selection: the method to select the value for each missing feature according to
          the distribution.
            - 'max': select the value with the highest probability.
            - 'sample': sample a value according to the distribution.
            - 'mean': select the mean of the distribution.
        eps: a small number to avoid numerical issues.

    Returns:
        y: the imputed data matrix.
    """
    assert arr.ndim == 2, "x must be a 2D matrix."
    assert arr.shape[1] == len(
        p_xgh
    ), "The number of features in x must be the same as the number of features in the\
          training data."
    assert (
        len(p_h) == p_xgh[0].shape[0]
    ), "The number of latent variables must be the same as the number of latent\
          variables in the training data."
    for col in range(arr.shape[1]):
        assert (
            np.unique(arr[:, col]).shape[0] <= num_categories[col]
        ), f"The number of categories of feature {col} in x must be less than or equal\
              to the number of categories in the training data."

    num_samples, num_feats = arr.shape
    output = []
    logp_h, logp_xgh = np.log(p_h + eps), [
        np.log(p_xgh[d] + eps) for d in range(num_feats)
    ]

    for record_idx in range(num_samples):
        missing_idx = np.where(np.isnan(arr[record_idx]))[0]
        if len(missing_idx) == 0:
            output.append(arr[record_idx])
            continue

        logpxm_given_xo = compute_log_prob(
            logp_h, logp_xgh, arr[record_idx], num_categories
        )

        # all possible combinations of missing features
        combs = np.array(
            np.meshgrid(*[np.arange(num_categories[m]).tolist() for m in missing_idx])
        ).T.reshape(-1, len(missing_idx))

        # log p(x_m|h) = log p(x1|h) + log p(x2|h) + ... = logsumexp_f log p(xf|h)
        logpxm_given_xo = logpxm_given_xo[:, np.newaxis].repeat(len(combs), 1)
        for i in range(len(combs)):
            for j, feat_idx in enumerate(missing_idx):
                logpxm_given_xo[:, i] += logp_xgh[feat_idx][:, combs[i, j]]

        logpxm_given_xo = special.logsumexp(logpxm_given_xo, axis=0)

        # normalize
        pxm_given_xo = condexp(logpxm_given_xo)

        imputed_record = arr[record_idx].copy()
        imputed_record[missing_idx] = predict_missing_values(
            pxm_given_xo, combs, missing_idx, selection
        )

        output.append(imputed_record)

    return np.array(output)


def get_model(args, num_output_classes, checkpoint=None):
    """
    Returns the model based on the provided arguments and optional checkpoint.

    Args:
        args: Argument parser object containing model configurations.
        num_output_classes: Number of output classes for the model.
        checkpoint: Optional checkpoint to load the model from.

    Returns:
        model: The initialized (and possibly pre-trained) PyTorch model.
    """
    # Initialize filter sizes
    num_filters = [args.n_base_filters * (2**i) for i in range(args.n_layers)]
    output_filters = vars(args).get("output_filters", 128)

    # Initialize the model
    model = Model(
        in_filters=args.in_filters,
        filters=num_filters,
        out_filters=output_filters,
        act=args.act,
        n_classes=num_output_classes,
        n_clinical_data=args.n_clinical_data,
    )
    model.to(args.device)

    # Load from checkpoint if provided
    if checkpoint is not None:
        new_state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(new_state_dict, strict=False)

        if args.n_clinical_data > 0:
            # Freeze the imaging branch (layers that exist in the checkpoint)
            for name, param in model.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False
        print("Model loaded from checkpoint.")

    # Log the number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable Args: {num_trainable_params}")

    return model


def model_to_ckpt(path):
    """
    Converts a model weights file to a checkpoint file.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = {}
    ckpt["model"] = torch.load(path, map_location=device)
    torch.save(ckpt, path)
