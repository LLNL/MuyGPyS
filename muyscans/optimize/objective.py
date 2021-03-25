#!/usr/bin/env python
# encoding: utf-8
"""
@file prediction.py

Created by priest2 on 2020-10-19

Leave-one-out hyperparameter optimization logic and testing.
"""

import numpy as np

from scipy.special import softmax
from sklearn.metrics import log_loss
from time import perf_counter


def get_loss_func(loss_method):
    loss_method = loss_method.lower()
    if loss_method == "cross-entropy" or loss_method == "log":
        return cross_entropy_fn
    elif loss_method == "mse":
        return mse_fn
    else:
        raise NotImplementedError(
            f"Loss function {loss_method} is not implemented."
        )


def cross_entropy_fn(predictions, targets):
    """
    Computes the cross entropy loss the predicted versus known response.
    Transforms `predictions' to be row-stochastic, and ensures that `targets'
    contains no negative elements.

    Parameters
    ----------
    predictions : numpy.ndarray(int), shape = ``(batch_count, response_count)''
        The predicted response.
    targets : numpy.ndarray(int), shape = ``(batch_count, response_count)''
        The expected response.

    Returns
    -------
    float
        The cross-entropy loss of the prediction.
    """
    one_hot_targets = np.zeros(targets.shape)
    one_hot_targets[targets > 0.0] = 1.0

    return log_loss(
        one_hot_targets, softmax(predictions, axis=1), eps=1e-6, normalize=False
    )


def mse_fn(predictions, targets):
    """
    Computes mean squared error loss of the predicted versus known response.
    Treats multivariate outputs as interchangeable in terms of loss penalty.

    Parameters
    ----------
    predictions : numpy.ndarray(int), shape = ``(batch_count, response_count)''
        The predicted response.
    targets : numpy.ndarray(int), shape = ``(batch_count, response_count)''
        The expected response.

    Returns
    -------
    float
        The mse loss of the prediction.
    """
    batch_count = predictions.shape[0]
    response_count = predictions.shape[1]
    squared_errors = np.sum((predictions - targets) ** 2)
    return squared_errors / (batch_count * response_count)


def loo_crossval(
    x0,
    objective_fn,
    muygps,
    params,
    batch_indices,
    batch_nn_indices,
    embedded_train,
    train_targets,
):
    """
    Returns leave-one-out cross validation performance for a `MuyGPS` object.
    Predicts on all of the training data at once.

    Parameters
    ----------
    x0 : float
        Hyperparameter values.
    objective_fn : callable
        The function to be used to optimize ``nu''.
    muygps : muyscans.GP.MuyGPS
        Local kriging approximate MuyGPS.
    params : set
        Set of parameter names to optimize.
    batch_indices : numpy.ndarray(int), shape = ``(batch_size,)''
        Batch observation indices.
    batch_nn_indices : numpy.ndarray(int), shape = ``(n_batch, nn_count)''
        Indices of the nearest neighbors
    batch_nn_distances : numpy.ndarray(float),
                         shape = ``(batch_size, nn_count)''
        Distances from each batch observation to its nearest neighbors.
    embedded_train : numpy.ndarray(float),
                     shape = ``(train_count, embedding_dim)''
        The full embedded training data matrix.
    train_targets : numpy.ndarray(float),
                   shape = ``(train_count, response_count)''
        List of output response for all embedded data, e.g. one-hot class
        encodings for classification.

    Returns
    -------
    float
        The evaluation of ``objective_fn'' on the predicted versus expected
        response.
    """
    muygps.set_param_array(params, x0)

    predictions = muygps.regress(
        batch_indices,
        batch_nn_indices,
        embedded_train,
        embedded_train,
        train_targets,
    )

    targets = train_targets[batch_indices, :]

    return objective_fn(predictions, targets)
