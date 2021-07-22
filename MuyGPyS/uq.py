# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""Convenience functions for uncertainty quantification workflows.
"""

import numpy as np

from typing import Callable, List, Tuple, Union

from MuyGPyS.gp.muygps import MuyGPS


def train_two_class_interval(
    surrogate: MuyGPS,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train: np.ndarray,
    train_responses: np.ndarray,
    train_labels: np.ndarray,
    objective_fns: Union[List[Callable], Tuple[Callable, ...]],
) -> np.ndarray:
    """
    For 2-class classification problems, get estimate of the confidence
    interval scaling parameter.

    Args:
        surrogate:
            Surrogate regressor.
        batch_indices:
            Batch observation indices of shape `(batch_count)`.
        batch_nn_indices:
            Indices of the nearest neighbors of shape `(batch_count, nn_count)`.
        train:
            The full training data matrix of shape
            `(train_count, feature_count)`.
        train_responses:
            One-hot encoding of class labels for all training data of shape
            `(train_count, class_count)`.
        train_labels:
            List of class labels for all training data of shape
            `(train_count,)`.
        objective_fns:
            A collection of `objective_count` functions taking the four
            arguments bit masks alpha and beta - the type 1 and type 2 error
            counts at each grid location, respectively - and the numbers of
            correctly and incorrectly classified training examples. Each
            objective function effervesces a cutoff value to calibrate UQ
            for class decision-making.

    Returns:
        A vector of shape `(objective_count)` indicating the confidence interval
        scale parameter that minimizes each considered objective function.
    """
    targets = train_labels[batch_indices]
    mean, variance = surrogate.regress_from_indices(
        batch_indices,
        batch_nn_indices,
        train,
        train,
        train_responses,
        variance_mode="diagonal",
    )
    predicted_labels = 2 * np.argmax(mean, axis=1) - 1

    correct_mask = predicted_labels == targets
    incorrect_mask = np.invert(correct_mask)

    # NOTE[bwp]: might want to make this range configurable by the user as well.
    cutv = np.linspace(0.01, 20, 1999)
    _alpha = np.zeros((len(cutv)))
    _beta = np.zeros((len(cutv)))
    for i in range(len(cutv)):
        _alpha[i] = 1 - np.mean(
            np.logical_and(
                (
                    mean[incorrect_mask, 1]
                    - cutv[i] * np.sqrt(variance[incorrect_mask])
                )
                < 0.0,
                (
                    mean[incorrect_mask, 1]
                    + cutv[i] * np.sqrt(variance[incorrect_mask])
                )
                > 0.0,
            )
        )
        _beta[i] = np.mean(
            np.logical_and(
                (
                    mean[correct_mask, 1]
                    - cutv[i] * np.sqrt(variance[correct_mask])
                )
                < 0.0,
                (
                    mean[correct_mask, 1]
                    + cutv[i] * np.sqrt(variance[correct_mask])
                )
                > 0.0,
            )
        )

    correct_count = np.sum(correct_mask)
    incorrect_count = np.sum(incorrect_mask)
    cutoffs = np.array(
        [
            cutv[obj_f(_alpha, _beta, correct_count, incorrect_count)]
            for obj_f in objective_fns
        ]
    )
    return cutoffs
