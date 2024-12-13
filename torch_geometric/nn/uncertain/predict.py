from typing import Optional, Tuple

import numpy as np


def tps(
    cal_smx: np.ndarray,
    val_smx: np.ndarray,
    cal_labels: np.ndarray,
    val_labels: np.ndarray,
    n: int,
    alpha: float,
) -> Tuple[np.ndarray, float, float]:
    """Top Prediction Set method for constructing prediction sets.

    :param cal_smx: Softmax probabilities for calibration set
    :type cal_smx: np.ndarray
    :param val_smx: Softmax probabilities for validation set
    :type val_smx: np.ndarray
    :param cal_labels: True labels for calibration set
    :type cal_labels: np.ndarray
    :param val_labels: True labels for validation set
    :type val_labels: np.ndarray
    :param n: Number of calibration samples
    :type n: int
    :param alpha: Desired miscoverage level (e.g., 0.1 for 90% coverage)
    :type alpha: float
    :return: Tuple containing (prediction sets, coverage, efficiency)
    :rtype: Tuple[np.ndarray, float, float]
    """
    # Compute nonconformity scores as complement of probability of true class
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]

    # Find score threshold for desired coverage
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, interpolation="higher")

    # Construct prediction sets
    prediction_sets = val_smx >= (1 - qhat)

    # Calculate metrics
    coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]),
        val_labels,
    ].mean()
    efficiency = np.sum(prediction_sets) / len(prediction_sets)

    return prediction_sets, coverage, efficiency


def aps(
    cal_smx: np.ndarray,
    val_smx: np.ndarray,
    cal_labels: np.ndarray,
    val_labels: np.ndarray,
    n: int,
    alpha: float,
) -> Tuple[np.ndarray, float, float]:
    """Adaptive Prediction Set method for constructing prediction sets.

    Constructs prediction sets by adaptively including labels based on
    cumulative probabilities until reaching the desired coverage level.

    :param cal_smx: Softmax probabilities for calibration set
    :type cal_smx: np.ndarray
    :param val_smx: Softmax probabilities for validation set
    :type val_smx: np.ndarray
    :param cal_labels: True labels for calibration set
    :type cal_labels: np.ndarray
    :param val_labels: True labels for validation set
    :type val_labels: np.ndarray
    :param n: Number of calibration samples
    :type n: int
    :param alpha: Desired miscoverage level (e.g., 0.1 for 90% coverage)
    :type alpha: float
    :return: Tuple containing (prediction sets, coverage, efficiency)
    :rtype: Tuple[np.ndarray, float, float]
    """
    # Sort probabilities in descending order and compute cumulative sums
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)

    # Get cumulative probability up to true label for each calibration example
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1),
                                    axis=1)[range(n), cal_labels]

    # Find score threshold for desired coverage
    qhat = np.quantile(cal_scores,
                       np.ceil((n + 1) * (1 - alpha)) / n,
                       interpolation="higher")

    # Construct prediction sets for validation data
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat,
                                         val_pi.argsort(axis=1), axis=1)

    # Calculate metrics
    coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]),
        val_labels,
    ].mean()
    efficiency = np.sum(prediction_sets) / len(prediction_sets)

    return prediction_sets, coverage, efficiency


def raps(
    cal_smx: np.ndarray,
    val_smx: np.ndarray,
    cal_labels: np.ndarray,
    val_labels: np.ndarray,
    n: int,
    alpha: float,
    lam_reg: float = 0.01,
    k_reg: Optional[int] = None,
    rand: bool = True,
    disallow_zero_sets: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """Regularized Adaptive Prediction Set method.

    A regularized version of APS that penalizes large prediction sets
    and includes randomization for improved calibration.

    :param cal_smx: Softmax probabilities for calibration set
    :type cal_smx: np.ndarray
    :param val_smx: Softmax probabilities for validation set
    :type val_smx: np.ndarray
    :param cal_labels: True labels for calibration set
    :type cal_labels: np.ndarray
    :param val_labels: True labels for validation set
    :type val_labels: np.ndarray
    :param n: Number of calibration samples
    :type n: int
    :param alpha: Desired miscoverage level
    :type alpha: float
    :param lam_reg: Regularization strength
    :type lam_reg: float
    :param k_reg: Number of top classes to leave unpenalized
    :type k_reg: Optional[int]
    :param rand: Whether to use randomization
    :type rand: bool
    :param disallow_zero_sets: Whether to force including at least one class
    :type disallow_zero_sets: bool
    :return: Tuple containing (prediction sets, coverage, efficiency)
    :rtype: Tuple[np.ndarray, float, float]
    """
    # Set default k_reg if not provided
    if k_reg is None:
        k_reg = min(5, cal_smx.shape[1])

    # Create regularization vector: no penalty for top k_reg classes
    reg_vec = np.array(k_reg * [
        0,
    ] + (cal_smx.shape[1] - k_reg) * [
        lam_reg,
    ])[None, :]

    # Process calibration data
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:, None])[1]

    # Compute randomized scores for calibration set
    rand_term = np.random.rand(n) * cal_srt_reg[np.arange(n),
                                                cal_L] if rand else 0
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - rand_term

    # Find score threshold for desired coverage
    qhat = np.quantile(cal_scores,
                       np.ceil((n + 1) * (1 - alpha)) / n,
                       interpolation="higher")

    # Process validation data
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
    val_srt_reg = val_srt + reg_vec

    # Construct prediction sets
    if rand:
        rand_term = np.random.rand(n_val, 1) * val_srt_reg
        indicators = (val_srt_reg.cumsum(axis=1) - rand_term) <= qhat
    else:
        indicators = val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat

    if disallow_zero_sets:
        indicators[:, 0] = True

    prediction_sets = np.take_along_axis(
        indicators,
        val_pi.argsort(axis=1),
        axis=1,
    )

    # Calculate metrics
    coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]),
        val_labels,
    ].mean()
    efficiency = np.sum(prediction_sets) / len(prediction_sets)

    return prediction_sets, coverage, efficiency


def cqr(
    cal_labels: np.ndarray,
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    val_labels: np.ndarray,
    val_lower: np.ndarray,
    val_upper: np.ndarray,
    n: int,
    alpha: float,
) -> Tuple[list, float, float]:
    """Conformalized Quantile Regression (CQR).

    Constructs prediction intervals using conformalized quantile regression.
    """
    # Compute nonconformity scores
    cal_scores = np.maximum(cal_labels - cal_upper, cal_lower - cal_labels)

    # Find score threshold for desired coverage
    qhat = np.quantile(cal_scores,
                       np.ceil((n + 1) * (1 - alpha)) / n,
                       interpolation="higher")

    # Construct prediction intervals
    prediction_sets = [val_lower - qhat, val_upper + qhat]

    # Calculate metrics
    coverage = ((val_labels >= prediction_sets[0]) &
                (val_labels <= prediction_sets[1])).mean()
    efficiency = np.mean(val_upper + qhat - (val_lower - qhat))

    return prediction_sets, coverage, efficiency


def qr(
    cal_labels: np.ndarray,
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    val_labels: np.ndarray,
    val_lower: np.ndarray,
    val_upper: np.ndarray,
    n: int,
    alpha: float,
) -> Tuple[list, float, float]:
    """Simple Quantile Regression (QR) for regression tasks.

    Returns prediction intervals directly from quantile predictions
    without conformalization.
    """
    prediction_sets = [val_lower, val_upper]
    coverage = ((val_labels >= prediction_sets[0]) &
                (val_labels <= prediction_sets[1])).mean()
    efficiency = np.mean(val_upper - val_lower)
    return prediction_sets, coverage, efficiency
