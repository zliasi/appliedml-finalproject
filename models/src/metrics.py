"""Standalone metric functions for model evaluation."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mean_absolute_error(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> float:
    """Compute R-squared (Coefficient of Determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R2 value (1.0 = perfect, 0.0 = mean prediction)
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
