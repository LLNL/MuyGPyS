# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Tensor functions

Compute pairwise and crosswise difference tensors for the purposes of kernel
construction.

See the following example computing the pairwise and crosswise differences
between a batch of training data and their nearest neighbors.

Example:
    >>> from MuyGPyS.neighbors import NN_Wrapper
    >>> from MuyGPyS.optimize.batch import sample_batch
    >>> from MuyGPyS.gp.tensors import crosswise_tensor, pairwise_tensor
    >>> train_features = load_train_features()
    >>> nn_count = 10
    >>> nbrs_lookup = NN_Wrapper(
    ...         train_features,
    ...         nn_count,
    ...         nn_method="exact",
    ...         algorithm="ball_tree",
    ... )
    >>> train_count, _ = train_features.shape
    >>> batch_count = 50
    >>> batch_indices, batch_nn_indices = sample_batch(
    ...         nbrs_lookup, batch_count, train_count
    ... )
    >>> pairwise_diffs = pairwise_tensor(
    ...         train_features, batch_nn_inidices
    ... )
    >>> crosswise_diffs = crosswise_tensor(
    ...         train_features,
    ...         train_features,
    ...         batch_indices,
    ...         batch_nn_indices,
    ... )

See also the following example computing the crosswise differences between a
test dataset and their nearest neighors in the training data.

Example:
    >>> from MuyGPyS.neighbors import NN_Wrapper
    >>> from MuyGPyS.gp.tensors import crosswise_tensor, pairwise_tensor
    >>> train_features = load_train_features()
    >>> test_features = load_test_features()
    >>> nn_count = 10
    >>> nbrs_lookup = NN_Wrapper(
    ...         train_features, nn_count, nn_method="exact", algorithm="ball_tree"
    ... )
    >>> nn_indices, nn_diffs = nbrs_lookup.get_nns(test_features)
    >>> test_count, _ = test_features.shape
    >>> indices = np.arange(test_count)
    >>> nn_indices, _ = nbrs_lookup.get_nns(test_features)
    >>> pairwise_diffs = pairwise_tensor(
    ...         train_features, nn_inidices
    ... )
    >>> crosswise_diffs = crosswise_tensor(
    ...         test_features,
    ...         train_features,
    ...         indices,
    ...         nn_indices,
    ... )

The helper functions :func:`MuyGPyS.gp.tensors.make_predict_tensors`,
:func:`MuyGPyS.gp.tensors.make_fast_predict_tensors`, and
:func:`MuyGPyS.gp.tensors.make_train_tensors` wrap these difference tensors and
also return the nearest neighbors sets' training targets and (in the latter
case) the training targets of the training batch. These functions are convenient
as the difference and target tensors are usually needed together.
"""

from typing import Optional, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import (
    _make_fast_predict_tensors,
    _make_predict_tensors,
    _make_train_tensors,
    _batch_features_tensor,
    _crosswise_tensor,
    _pairwise_tensor,
    _fast_nn_update,
    _make_heteroscedastic_tensor,
)


def make_heteroscedastic_tensor(
    measurement_noise: mm.ndarray,
    batch_nn_indices: mm.ndarray,
) -> mm.ndarray:
    """
    Create the heteroscedastic noise tensor for nonuniform noise values.

    Used to produce the noise tensor needed during batched training and
    prediction. Creates the `noise_tensor` tensor required by heteroscedastic
    MuyGPs models.

    Args:
        measurement_noise:
            A matrix of floats of shape `(batch_count,)` providing the noise
            corresponding to the response variable at each input value in the
            data.
        batch_nn_indices:
            A matrix of integers of shape `(batch_count, nn_count, nn_count)`
            listing the measurement noise for the nearest neighbors for all
            observations in the batch.

    Returns:
        A matrix of floats of shape `(batch_count, nn_count)` providing the
        noise corresponding to the nearest neighbor responses for all
        observations in the batch.
    """
    return _make_heteroscedastic_tensor(measurement_noise, batch_nn_indices)


def fast_nn_update(
    train_nn_indices: mm.ndarray,
) -> mm.ndarray:
    """
    Modify the nearest neighbor indices of the training data to include self.

    This function is only intended for use in concert with
    :func:~MuyGPyS.gp.tensors.make_fast_predict_tensors` and
    :func:`MuyGPyS.gp.muygps.MuyGPS.fast_coefficients`.

    Example:
        >>> train_nn_indices, _ = nbrs_lookup.get_nns(train_features)
        >>> train_nn_indices = fast_nn_update(train_nn_indices)
        >>> pairwise_diffs, nn_targets = make_fast_predict_tensors(
        ...     train_nn_indices,
        ...     train_features,
        ...     train_responses,
        ... )
        >>> K = muygps_fast.kernel(pairwise_diffs)
        >>> precomputed_coefficients_matrix = muygps_fast.fast_coefficients(
        ...     K, nn_targets
        ... )
        >>> # Late on, once test data is encountered
        >>> test_indices = np.arange(test_count)
        >>> test_nn_indices, _ = nbrs_lookup.get_nns(test_features)
        >>> closest_neighbor = test_nn_indices[:, 0]
        >>> closest_set = train_nn_indices[closest_neighbor, :]

    Args:
        train_nn_indices:
            A matrix of integers of shape `(train_count, nn_count)` listing the
            nearest neighbor indices for all observations in the batch.

    Returns:
        An integral matrix of shape `(train_count, nn_count)` that is similar
        to the input, but the most distant neighbor index is removed and the
        index reference to self has been inserted.
    """
    return _fast_nn_update(train_nn_indices)


def make_fast_predict_tensors(
    batch_nn_indices: mm.ndarray,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Create the difference and target tensors for fast posterior mean inference.

    Creates `pairwise_diffs` and `batch_nn_targets` tensors required by
    :func:`MuyGPyS.gp.muygps.MuyGPS.fast_posterior_mean`.

    Args:
        batch_nn_indices:
            A matrix of integers of shape `(batch_count, nn_count)` listing the
            nearest neighbor indices for all observations in the batch.
        train_features:
            The full floating point training data matrix of shape
            `(train_count, feature_count)`.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows are
            vector-valued responses for each training element.

    Returns
    -------
    pairwise_diffs:
        A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
        containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
        nearest neighbor difference tensors corresponding to each of the
        batch elements.
    batch_nn_targets:
        Tensor of floats of shape `(batch_count, nn_count, response_count)`
        containing the expected response for each nearest neighbor of each batch
        element.
    """
    return _make_fast_predict_tensors(
        batch_nn_indices,
        train_features,
        train_targets,
    )


def make_predict_tensors(
    batch_indices: mm.ndarray,
    batch_nn_indices: mm.ndarray,
    test_features: Optional[mm.ndarray],
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray, mm.ndarray]:
    """
    Create the difference and target tensors for prediction.

    Creates the `crosswise_diffs`, `pairwise_diffs` and `batch_nn_targets`
    tensors required by :func:`~MuyGPyS.gp.MuyGPS.posterior_mean` and
    :func:`~MuyGPyS.gp.MuyGPS.posterior_variance`.

    Args:
        batch_indices:
            A vector of integers of shape `(batch_count,)` identifying the
            training batch of observations to be approximated.
        batch_nn_indices:
            A matrix of integers of shape `(batch_count, nn_count)` listing the
            nearest neighbor indices for all observations in the batch.
        test_features:
            The full floating point testing data matrix of shape
            `(test_count, feature_count)`.
        train_features:
            The full floating point training data matrix of shape
            `(train_count, feature_count)`.
        train_targets:
            A matrix of shape `(train_count, feature_count)` whose rows are
            vector-valued responses for each training element.

    Returns
    -------
    crosswise_diffs:
        A tensor of shape `(batch_count, nn_count, feature_count)` whose last
        two dimensions list the difference between each feature of each batch
        element element and its nearest neighbors.
    pairwise_diffs:
        A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
        containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
        nearest neighbor difference tensors corresponding to each of the
        batch elements.
    batch_nn_targets:
        Tensor of floats of shape `(batch_count, nn_count, response_count)`
        containing the expected response for each nearest neighbor of each batch
        element.
    """
    return _make_predict_tensors(
        batch_indices,
        batch_nn_indices,
        test_features,
        train_features,
        train_targets,
    )


def make_train_tensors(
    batch_indices: mm.ndarray,
    batch_nn_indices: mm.ndarray,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray, mm.ndarray, mm.ndarray]:
    """
    Create the difference and target tensors needed for training.

    Similar to :func:`~MuyGPyS.gp.data.make_predict_tensors` but returns the
    additional `batch_targets` matrix, which is only defined for a batch of
    training data.

    Args:
        batch_indices:
            A vector of integers of shape `(batch_count,)` identifying the
            training batch of observations to be approximated.
        batch_nn_indices:
            A matrix of integers of shape `(batch_count, nn_count)` listing the
            nearest neighbor indices for all observations in the batch.
        train_features:
            The full floating point training data matrix of shape
            `(train_count, feature_count)`.
        train_targets:
            A matrix of shape `(train_count, feature_count)` whose rows are
            vector-valued responses for each training element.

    Returns
    -------
    crosswise_diffs:
        A tensor of shape `(batch_count, nn_count, feature_count)` whose last
        two dimensions list the difference between each feature of each batch
        element element and its nearest neighbors.
    pairwise_diffs:
        A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
        containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
        nearest neighbor difference tensors corresponding to each of the batch
        elements.
    batch_targets:
        Matrix of floats of shape `(batch_count, response_count)` whose rows
        give the expected response for each batch element.
    batch_nn_targets:
        Tensor of floats of shape `(batch_count, nn_count, response_count)`
        containing the expected response for each nearest neighbor of each batch
        element.
    """
    return _make_train_tensors(
        batch_indices, batch_nn_indices, train_features, train_targets
    )


def batch_features_tensor(
    features: mm.ndarray,
    batch_indices: mm.ndarray,
) -> mm.ndarray:
    """
    Compute a tensor of feature vectors for each batch element.

    Args:
        features:
            The full floating point training or testing data matrix of shape
            `(train_count, feature_count)` or `(test_count, feature_count)`.
        batch_indices:
            A vector of integers of shape `(batch_count,)` identifying the
            training batch of observations to be approximated.

    Returns:
        A tensor of shape `(batch_count, feature_count)` containing
        the feature vectors for each batch element.
    """
    return _batch_features_tensor(features, batch_indices)


def crosswise_tensor(
    data: mm.ndarray,
    nn_data: mm.ndarray,
    data_indices: mm.ndarray,
    nn_indices: mm.ndarray,
) -> mm.ndarray:
    """
    Compute a matrix of differences between data and their nearest neighbors.

    Takes full datasets of records of interest `data` and neighbor candidates
    `nn_data` and produces the differences between each element of `data`
    indicated by `data_indices` and each of the nearest neighbors
    in `nn_data` as indicated by the corresponding rows of `nn_indices`. `data`
    and `nn_data` can refer to the same dataset.

    See the following example computing the crosswise differences between a
    batch of training data and their nearest neighbors.

    Args:
        data:
            The data matrix of shape `(data_count, feature_count)` containing
            batch elements.
        nn_data:
            The data matrix of shape `(candidate_count, feature_count)`
            containing the universe of candidate neighbors for the batch
            elements. Might be the same as `data`.
        indices:
            An integral vector of shape `(batch_count,)` containing the indices
            of the batch.
        nn_indices:
            An integral matrix of shape (batch_count, nn_count) listing the
            nearest neighbor indices for the batch of data points.

    Returns:
        A tensor of shape `(batch_count, nn_count, feature_count)` whose last
        two dimensions list the difference between each feature of each batch
        element element and its nearest neighbors.
    """
    return _crosswise_tensor(data, nn_data, data_indices, nn_indices)


def pairwise_tensor(
    data: mm.ndarray,
    nn_indices: mm.ndarray,
) -> mm.ndarray:
    """
    Compute a tensor of pairwise differences among sets of nearest neighbors.

    Takes a full dataset of records of interest `data` and produces the
    pairwise differences between the elements indicated by each row of
    `nn_indices`.

    Args:
        data:
            The data matrix of shape `(batch_count, feature_count)` containing
            batch elements.
        nn_indices:
            An integral matrix of shape (batch_count, nn_count) listing the
            nearest neighbor indices for the batch of data points.

    Returns:
        A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
        containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
        nearest neighbor difference tensors corresponding to each of the
        batch elements.
    """
    return _pairwise_tensor(data, nn_indices)
