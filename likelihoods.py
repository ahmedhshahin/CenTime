"""
Survival Analysis Utilities

This module contains various utility functions, distribution functions,
and loss functions for survival analysis. It includes implementations
for classical loss, centime loss, Cox loss, and DeepHit loss.

Author: Ahmed H. Shahin
Date: 31/8/2023
"""

import torch
import numpy as np
from torch import Tensor


# Helper Functions
def _calculate_distribution(
    loc: Tensor, var: Tensor, tmax: int, distribution_fn, return_prob: bool = False
) -> Tensor:
    """
    Helper function to get either log probabilities or probabilities.

    Args:
        loc (Tensor): Location parameter tensor.
        var (Tensor): Variance parameter tensor.
        tmax (int)  : Maximum time.
        distribution_fn (function): Distribution function to use.
        return_prob (bool, optional): Flag to return probabilities.
                                     Defaults to False.

    Returns:
        Tensor: Log probabilities or probabilities.
    """
    loc = loc.view(-1, 1)
    var = var.view(-1, 1)
    return distribution_fn(loc, var, tmax, return_prob)


# Distribution Functions
def discretized_logistic(
    loc: Tensor, var: Tensor, tmax: int, return_prob: bool = False
) -> Tensor:
    """
    Calculate discretized logistic distribution.

    Args:
        loc (Tensor): Location parameter tensor, shape (batch_size, 1).
        var (Tensor): Variance parameter tensor, shape (batch_size, 1).
        tmax (int): Maximum time.
        return_prob (bool, optional): Flag to return probabilities.
                                      Defaults to False.

    Returns:
        Tensor: Log probabilities or probabilities.
    """
    scale = torch.sqrt(var)
    trange = torch.arange(1, float(tmax) + 1, device=loc.device)
    probs = torch.sigmoid((trange + 0.5 - loc) / scale) - torch.sigmoid(
        (trange - 0.5 - loc) / scale
    )
    probs /= probs.sum(dim=1, keepdim=True)
    return torch.log(probs + 1e-8) if not return_prob else probs


def discretized_gaussian(
    loc: Tensor, var: Tensor, tmax: int, return_prob: bool = False
) -> Tensor:
    """
    Calculate discretized Gaussian distribution.

    Args:
        loc (Tensor): Location parameter tensor, shape (batch_size, 1).
        var (Tensor): Variance parameter tensor, shape (batch_size, 1).
        tmax (int): Maximum time.
        return_prob (bool, optional): Flag to return probabilities.
                                      Defaults to False.

    Returns:
        Tensor: Log probabilities or probabilities.
    """
    trange = torch.arange(1, float(tmax) + 1, device=loc.device)
    log_probs = -((trange - loc) ** 2) / (2 * var + 1e-8)
    log_probs -= torch.logsumexp(log_probs, dim=1, keepdim=True)
    return log_probs if not return_prob else torch.exp(log_probs)


# Utility Functions
def get_survival_function(prob: Tensor) -> np.ndarray:
    """
    Calculate survival function.

    Args:
        p (Tensor): Probability tensor, shape (batch_size, tmax).

    Returns:
        np.ndarray: Survival function values.
    """
    prob = prob.data.cpu().numpy()
    return np.cumsum(prob[:, ::-1], axis=1)[:, ::-1]


def get_mean_prediction(prob: Tensor, tmax: int) -> Tensor:
    """
    Calculate mean prediction.

    Args:
        p (Tensor): Probability tensor, shape (batch_size, tmax).
        tmax (int): Maximum time.

    Returns:
        Tensor: Mean prediction values.
    """
    time_range = torch.arange(1, tmax + 1, device=prob.device)
    return (time_range * prob).sum(dim=1).data


def get_probs(
    loc: Tensor,
    variance: Tensor,
    tmax: int,
    distribution: str = "discretized_gaussian",
) -> Tensor:
    """
    Get probabilities based on the distribution.

    Args:
        loc (Tensor): loc parameter tensor, shape (batch_size, 1).
        variance (Tensor): Variance parameter tensor, shape (batch_size, 1).
        tmax (int): Maximum time.
        distribution (str, optional): Type of distribution to use.
                                      Defaults to 'discretized_gaussian'.

    Returns:
        Tensor: Probabilities.
    """
    distribution_fn = (
        discretized_gaussian
        if distribution == "discretized_gaussian"
        else discretized_logistic
    )
    return _calculate_distribution(
        loc, variance, tmax, distribution_fn, return_prob=True
    )


# Loss Functions
def classical_loss(
    loc: Tensor,
    var: Tensor,
    delta: Tensor,
    time: Tensor,
    tmax: int,
    distribution: str = "discretized_gaussian",
) -> (Tensor, Tensor):
    """
    Compute classical loss for survival analysis.

    Args:
        loc (Tensor): Location parameter tensor, shape (batch_size, 1).
        var (Tensor): Variance parameter tensor, shape (batch_size, 1).
        delta (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
                                    censoring time if censored.
        tmax (int): Maximum time.
        distribution (str, optional): Distribution to use.
                                    Defaults to 'discretized_gaussian'.

    Returns:
        Tuple[Tensor, Tensor]: Loss for censored and uncensored data.
    """
    time = time.view(-1)
    delta = delta.view(-1).type(torch.bool)
    uncens_time = time[delta].view(-1, 1)
    cens_time = time[~delta].view(-1, 1)

    logp = _calculate_distribution(
        loc,
        var,
        tmax,
        discretized_gaussian
        if distribution == "discretized_gaussian"
        else discretized_logistic,
    )
    logp_cens = logp[~delta]
    logp_uncens = logp[delta]
    loss_uncens = logp_uncens.gather(
        1, uncens_time - 1
    ).sum()  # Adjust for 0-based indexing

    loss_cens = 0
    for c_time, _logp in zip(cens_time, logp_cens):
        loss_cens += torch.logsumexp(_logp[c_time:], 0)
    loss_cens = -loss_cens / len(time)

    loss_uncens = -loss_uncens / len(time)
    return loss_cens, loss_uncens


def centime_loss(
    loc: Tensor,
    var: Tensor,
    delta: Tensor,
    time: Tensor,
    tmax: int,
    distribution: str = "discretized_gaussian",
) -> (Tensor, Tensor):
    """
    Compute the centime loss for survival analysis.

    Args:
        loc (Tensor): Location parameter tensor, shape (batch_size, 1).
        var (Tensor): Variance parameter tensor, shape (batch_size, 1).
        delta (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
                                    censoring time if censored.
        tmax (int): Maximum time.
        distribution (str, optional): Distribution to use.
                                    Defaults to 'discretized_gaussian'.

    Returns:
        Tuple[Tensor, Tensor]: Loss for censored and uncensored data.
    """
    time = time.view(-1)
    delta = delta.view(-1).type(torch.bool)
    uncens_time = time[delta].view(-1, 1)
    cens_time = time[~delta].view(-1, 1)

    logp = _calculate_distribution(
        loc,
        var,
        tmax,
        discretized_gaussian
        if distribution == "discretized_gaussian"
        else discretized_logistic,
    )
    logp_cens = logp[~delta]
    logp_uncens = logp[delta]
    loss_uncens = logp_uncens.gather(
        1, uncens_time - 1
    ).sum()  # Adjust for 0-based indexing

    loss_cens = 0
    for c_time, _logp in zip(cens_time, logp_cens):
        factor = torch.arange(c_time[0], float(tmax), device=loc.device).log()
        loss_cens += torch.logsumexp(_logp[c_time:] - factor, 0)
    loss_cens = -loss_cens / len(time)

    loss_uncens = -loss_uncens / len(time)
    return loss_cens, loss_uncens


def cox_loss(theta: Tensor, delta: Tensor, time: Tensor) -> Tensor:
    """
    Compute Cox proportional hazards loss.

    Args:
        theta (Tensor): Theta tensor, shape (batch_size, 1).
        delta (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
                                censoring time if censored.

    Returns:
        Tensor: Cox loss.
    """
    if not is_valid_cox_batch(delta, time):
        return theta.sum() * 0

    time = time.reshape(-1)
    theta = theta.reshape(-1, 1)
    risk_mat = (time >= time[:, None]).float()
    loss_cox = (
        theta.reshape(-1) - logsumexp(theta.T, mask=risk_mat, dim=1)
    ) * delta.reshape(-1)
    loss_cox = loss_cox.sum() / delta.sum()
    return -loss_cox


def deephit_loss(
    pred: Tensor, event: Tensor, time: Tensor, sigma: float = 0.1, ranking: bool = False
) -> Tensor:
    """
    Compute DeepHit loss.

    Args:
        pred (Tensor): Prediction tensor, shape (batch_size, tmax).
        event (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
                                censoring time if censored.
        sigma (float, optional): Hyper-parameter for ranking loss.
                                Defaults to 0.1.
        ranking (bool, optional): Flag to include ranking term.
                                Defaults to False.

    Returns:
        Tensor: DeepHit loss.
    """
    if ranking:
        return deephit_likelihood(pred, event, time) + deephit_ranking(
            pred, event, time, sigma
        )
    return deephit_likelihood(pred, event, time)


def deephit_likelihood(pred: Tensor, event: Tensor, time: Tensor) -> Tensor:
    """
    Compute DeepHit likelihood objective.

    Args:
        pred (Tensor): Prediction tensor, shape (batch_size, tmax).
        event (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
          censoring time if censored.

    Returns:
        Tensor: DeepHit likelihood.
    """
    eps = 1e-8
    cif = pred.cumsum(1).gather(1, time)  # Cumulative incidence function
    loss_uncensored = (event * pred.gather(1, time)).add(eps).log().sum()
    loss_censored = ((1 - event) * (1 - cif)).add(eps).log().sum()

    loss_uncensored /= event.sum() if event.sum() > 0 else 1
    loss_censored /= (1 - event).sum() if (1 - event).sum() > 0 else 1
    return -(loss_uncensored + loss_censored)


def deephit_ranking(
    pred: Tensor, event: Tensor, time: Tensor, sigma: float = 0.1
) -> Tensor:
    """
    Compute DeepHit ranking term.

    Args:
        pred (Tensor): Prediction tensor, shape (batch_size, tmax).
        event (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
          censoring time if censored.
        sigma (float, optional): Hyper-parameter for ranking loss.
                                 Defaults to 0.1.

    Returns:
        Tensor: DeepHit ranking term.
    """
    prior_events_mask = (time < time[:, 0]).float() * event
    cdf_difference = _diff_cdf_at_time_i(pred, time)
    eta = torch.exp(-cdf_difference / sigma)
    loss = (prior_events_mask * eta).mean() if prior_events_mask.sum() > 0 else 0
    return loss


def _diff_cdf_at_time_i(pmf: Tensor, time: Tensor) -> Tensor:
    """
    Compute the difference in CDF between individual i and j at the event time of j.
    From:
      https://github.com/havakv/pycox/blob/0e9d6f9a1eff88a355ead11f0aa68bfb94647bf8/pycox/models/loss.py#L176
    R is the matrix from the DeepHit code giving the difference in CDF between
      individual i and j, at the event time of j.
        I.e: R_ij = F_i(T_i) - F_j(T_i)

    Args:
        pmf (Tensor): Matrix with probability mass function pmf_ij = f_i(t_j)
          (pred in our notation), shape (batch_size, tmax).
        time (Tensor): Matrix with indicator of duration/censor time.
          (time in our notation), shape (batch_size, 1).

    Returns:
        Tensor: Difference in CDF. R_ij = F_i(T_i) - F_j(T_i).
    """
    batch_size = pmf.shape[0]
    ones = torch.ones((batch_size, 1), device=pmf.device)
    time = time.reshape(-1, 1)
    # the code below assumes time is one hot, so we will convert it
    time = torch.zeros_like(pmf).scatter(1, time, 1.0)
    cdf_difference_matrix = pmf.cumsum(1).matmul(time.transpose(0, 1))
    diag_r = cdf_difference_matrix.diag().view(1, -1)
    cdf_difference_matrix = ones.matmul(diag_r) - cdf_difference_matrix
    return cdf_difference_matrix.transpose(0, 1)


def is_valid_cox_batch(delta: Tensor, time: Tensor) -> bool:
    """
    Check if the batch is valid for computing Cox loss.
    In the cases below, loss is not defined
    1. If there is no uncensored data
    2. If there are uncensored samples but the risk matrix is empty.
      I.e., no censored patients survived more that the event time of any
        uncensored patient.

    Args:
        delta (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Survival time tensor, shape (batch_size, 1).

    Returns:
        bool: True if valid, False otherwise.
    """
    risk_matrix = (time >= time[:, None]).float()
    return ((risk_matrix.sum(dim=1) * delta.reshape(-1)) > 1).any()


def logsumexp(
    input_tensor: Tensor, mask: Tensor = None, dim: int = None, keepdim: bool = False
) -> Tensor:
    """
    Compute the log of the sum of exponentials of input elements (masked).

    Args:
        input_tensor (Tensor): Input tensor.
        mask (Tensor, optional): Mask tensor, same shape as x.
        dim (int, optional): Dimension to reduce.
        keepdim (bool, optional): Keep dimension.

    Returns:
        Tensor: Result tensor.
    """
    if dim is None:
        input_tensor, dim = input_tensor.view(-1), 0
    max_value, _ = torch.max(input_tensor, dim=dim, keepdim=True)
    input_tensor = input_tensor - max_value
    res = torch.exp(input_tensor)
    if mask is not None:
        res = res * mask
    res = torch.log(torch.sum(res, dim=dim, keepdim=keepdim) + 1e-8)
    return res + max_value.squeeze(dim) if not keepdim else res + max_value
