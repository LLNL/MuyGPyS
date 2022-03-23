# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from scipy.special import softmax
from sklearn.metrics import log_loss


def _cross_entropy_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
    ll_eps: float = 1e-15,
) -> float:
    one_hot_targets = np.where(targets > 0.0, 1.0, 0.0)
    softmax_predictions = softmax(predictions, axis=1)

    return log_loss(
        one_hot_targets, softmax_predictions, eps=ll_eps, normalize=False
    )


def _mse_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    batch_count, response_count = predictions.shape
    squared_errors = np.sum((predictions - targets) ** 2)
    return squared_errors / (batch_count * response_count)
