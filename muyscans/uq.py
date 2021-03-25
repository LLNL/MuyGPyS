#!/usr/bin/env python
# encoding: utf-8
"""
@file uq.py

Created by priest2 on 2020-10-19

Leave-one-out hyperparameter optimization logic and testing.
"""

import numpy as np


def train_two_class_interval(
    muygps,
    batch_indices,
    batch_nn_indices,
    train,
    train_labels,
    train_lookup,
    objective_fns,
):
    """
    For 2-class classification problems, get estimate of the confidence
    interval scaling parameter.

    Parameters
    ----------
    muygps : muyscans.GP.MuyGPS
        Local kriging approximate MuyGPS.
    batch_indices : numpy.ndarray(int), shape = ``(batch_size,)''
        Batch observation indices.
    batch_nn_indices : numpy.ndarray(int), shape = ``(n_batch, nn_count)''
        Indices of the nearest neighbors
    train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        The full training data matrix.
    train_labels : numpy.ndarray(int), shape = ``(train_count, class_count)''
        One-hot encoding of class labels for all training data.
    train_lookup : numpy.ndarray(int), shape = ``(train_count,)''
        List of class labels for all training data.
    objective_fns : list(Callable), shape = ``(objective_count)''
        List of functions taking four arguments: bit masks alpha and beta - the
        type 1 and type 2 error counts at each grid location, respectively - and
        the numbers of correctly and incorrectly classified training examples.
        Each objective function effervesces a cutoff value that

    Returns
    -------
    cutoffs : np.ndarray(float), shape = ``(objective_count,)''
        Returns the confidence interval scale parameter that minimizes each
        considered objective function.
    """
    batch_count = batch_nn_indices.shape[0]
    train_count = train_lookup.shape[0]
    nn_count = batch_nn_indices.shape[1]

    targets = train_lookup[batch_indices]
    mean, variance = muygps.regress(
        batch_indices,
        batch_nn_indices,
        train,
        train,
        train_labels,
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
