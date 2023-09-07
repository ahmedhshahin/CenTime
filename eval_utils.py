"""
Evaluation metrics for survival analysis.

Includes implementations of concordance index, mean absolute error, and relative
 absolute error.

Author: Ahmed H. Shahin
Date: 31/8/2023
"""

import numpy as np
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import concordance_index_censored


def get_median_survival_cox(
    tr_preds: np.ndarray,
    tr_events: np.ndarray,
    tr_times: np.ndarray,
    ts_preds: np.ndarray,
    tmax: int = 156,
) -> np.ndarray:
    """
    Compute the median survival time from the predicted risk scores using Cox model.

    Args:
        tr_preds (np.ndarray): Training set predicted risk scores. Shape: (n_samples,).
        tr_events (np.ndarray): Training set binary event labels. Shape: (n_samples,).
        tr_times (np.ndarray): Training set time-to-event labels. Shape: (n_samples,).
                                Event time if event is True, censoring time otherwise.
        ts_preds (np.ndarray): Test set predicted risk scores. Shape: (n_samples,).
        tmax (int, optional): Maximum time to consider. Defaults to 156 (tmax in OSIC
          dataset). Used as the upper bound of the time range, as the default behavior
            of the Breslow estimator is to output infinity if the survival value does
              not drop below 0.5.

    Returns:
        np.ndarray: Predicted median survival time for each sample in the test set.
    """

    breslow = BreslowEstimator().fit(tr_preds, tr_events, tr_times)
    min_time, max_time = tr_times.min(), tr_times.max()
    times = np.arange(min_time, max_time)

    sample_surv_fn = breslow.get_survival_function(ts_preds)
    intermediate_preds = np.asarray([[fn(t) for t in times] for fn in sample_surv_fn])

    preds = []
    for pred in intermediate_preds:
        median_time_idx = np.where(pred <= 0.5)[0]
        preds.append(median_time_idx[0] if len(median_time_idx) > 0 else tmax)

    return np.array(preds)


def cindex(time: np.ndarray, event: np.ndarray, risk_pred: np.ndarray) -> float:
    """
    Evaluates the concordance index based on the given predicted risk scores.

    Args:
        time (np.ndarray): Time. Event time if uncensored, censoring time otherwise.
          Shape: (n_samples,).
        event (np.ndarray): Event indicator. 1 if uncensored, 0 otherwise.
          Shape: (n_samples,).
        scores_pred (np.ndarray): Predicted risk/hazard scores. Shape: (n_samples,).

    Returns:
        float: The concordance index.
    """

    try:
        concordance_index = concordance_index_censored(event, time, risk_pred)[0]
    except ZeroDivisionError:
        print("Cannot divide by zero.")
        concordance_index = 0.5

    return concordance_index


def rae(
    t_pred: np.ndarray, t_true: np.ndarray, event: np.ndarray, mode: str = "uncens"
) -> float:
    """
    Calculates the Relative Absolute Error (RAE) for the given predicted and true
      time-to-event values.

    Args:
        t_pred (np.ndarray): Predicted time-to-event values. Shape: (n_samples,).
        t_true (np.ndarray): True time-to-event values. Shape: (n_samples,).
        event (np.ndarray): Event indicator. 1 if uncensored, 0 otherwise.
          Shape: (n_samples,).
        mode (str, optional): Specifies the mode of calculation. Can be either "uncens"
          for uncensored or "cens" for censored. Defaults to "uncens".

    Returns:
        float: The calculated Relative Absolute Error (RAE).

    Raises:
        AssertionError: If the mode is not one of ["uncens", "cens"].
    """

    assert mode in ["uncens", "cens"], "Invalid mode. Choose from ['uncens', 'cens']"

    if len(t_true) == 0:
        return 0.0

    abs_error_i = np.abs(t_pred - t_true)

    if mode == "uncens":
        error = abs_error_i[event]
        rel_error = np.divide(error, t_true[event])
        return np.sum(rel_error) / len(error)

    error = abs_error_i[~event]
    error = error[t_pred[~event] <= t_true[~event]]
    rel_error = np.divide(error, t_true[~event][t_pred[~event] <= t_true[~event]])

    return np.sum(rel_error) / (~event).sum()


def mae(
    t_pred: np.ndarray, t_true: np.ndarray, event: np.ndarray, mode: str = "uncens"
) -> float:
    """
    Calculates the Mean Absolute Error (MAE) for the given predicted and true
      time-to-event values.

    Args:
        t_pred (np.ndarray): Predicted time-to-event values. Shape: (n_samples,).
        t_true (np.ndarray): True time-to-event values. Shape: (n_samples,).
        event (np.ndarray): Event indicator. 1 if uncensored, 0 otherwise.
          Shape: (n_samples,).
        mode (str, optional): Specifies the mode of calculation. Can be either "uncens"
          for uncensored or "cens" for censored.
          Defaults to "uncens".

    Returns:
        float: The calculated Relative Absolute Error (RAE).

    Raises:
        AssertionError: If the mode is not one of ["uncens", "cens"].
    """

    assert mode in ["uncens", "cens"], "Invalid mode. Choose from ['uncens', 'cens']"

    if len(t_true) == 0:
        return 0.0

    abs_error = np.abs(t_pred - t_true)

    if mode == "uncens":
        error = abs_error[event]
        return np.sum(error) / len(error)

    error = abs_error[~event]
    error = error[t_pred[~event] <= t_true[~event]]

    return np.sum(error) / (~event).sum()
