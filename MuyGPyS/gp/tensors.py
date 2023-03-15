# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Distance functions

Compute pairwise and crosswise distance tensors for the purposes of kernel
construction.

See the following example computing the pairwise and crosswise distances between
a batch of training data and their nearest neighbors.

Example:
    >>> from MuyGPyS.neighbors import NN_Wrapper
    >>> from MuyGPyS.optimize.batch import sample_batch
    >>> from MuyGPyS.gp.tensors import crosswise_tensors, pairwise_tensors
    >>> train_features = load_train_features()
    >>> nn_count = 10
    >>> nbrs_lookup = NN_Wrapper(
    ...         train_features, nn_count, nn_method="exact", algorithm="ball_tree"
    ... )
    >>> train_count, _ = train_features.shape
    >>> batch_count = 50
    >>> batch_indices, batch_nn_indices = sample_batch(
    ...         nbrs_lookup, batch_count, train_count
    ... )
    >>> pairwise_diffs = pairwise_tensors(
    ...         train_features, batch_nn_inidices, metric="l2"
    ... )
    >>> crosswise_diffs = crosswise_tensors(
    ...         train_features,
    ...         train_features,
    ...         batch_indices,
    ...         batch_nn_indices,
    ...         metric="l2",
    ... )
    )

See also the following example computing the crosswise distances between a
test dataset and their nearest neighors in the training data.

Example:
    >>> from MuyGPyS.neighbors import NN_Wrapper
    >>> from MuyGPyS.gp.tensors import crosswise_tensors, pairwise_tensors
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
    >>> pairwise_diffs = pairwise_tensors(
    ...         train_features, nn_inidices, metric="l2"
    ... )
    >>> crosswise_diffs = crosswise_tensors(
    ...         test_features,
    ...         train_features,
    ...         indices,
    ...         nn_indices,
    ...         metric="l2"
    ... )

The helper functions :func:`MuyGPyS.gp.distance.make_predict_tensors`,
:func:`MuyGPyS.gp.distance.make_fast_predict_tensors`, and
:func:`MuyGPyS.gp.distance.make_train_tensors` wrap these distances tensors and
also return the nearest neighbors sets' training targets and (in the latter
case) the training targets of the training batch. These functions are convenient
as the distance and target tensors are usually needed together.
"""


from typing import Optional, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import (
    _make_fast_predict_tensors,
    _make_predict_tensors,
    _make_train_tensors,
    _crosswise_tensors,
    _pairwise_tensors,
    _fast_nn_update,
)


def fast_nn_update(
    batch_nn_indices: mm.ndarray,
) -> mm.ndarray:
    return _fast_nn_update(batch_nn_indices)


def make_fast_predict_tensors(
    metric: str,
    batch_nn_indices: mm.ndarray,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Create the distance and target tensors for fast posterior mean inference.

    Creates `pairwise_diffs` and `batch_nn_targets` tensors required by
    :func:`~MuyGPyS.gp.muygps.MuyGPS.fast_posterior_mean`.

    Args:
        metric:
            The metric to be used to compute distances.
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
            A matrix of shape `(train_count, response_count)` whose rows are
            vector-valued responses for each training element.

    Returns
    -------
    pairwise_diffs:
        A tensor of shape `(batch_count, nn_count, nn_count)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements.
    batch_nn_targets:
        Tensor of floats of shape `(batch_count, nn_count, response_count)`
        containing the expected response for each nearest neighbor of each batch
        element.
    """
    return _make_fast_predict_tensors(
        metric,
        batch_nn_indices,
        train_features,
        train_targets,
    )


def make_predict_tensors(
    metric: str,
    batch_indices: mm.ndarray,
    batch_nn_indices: mm.ndarray,
    test_features: Optional[mm.ndarray],
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray, mm.ndarray]:
    """
    Create the distance and target tensors for prediction.

    Creates the `crosswise_diffs`, `pairwise_diffs` and `batch_nn_targets`
    tensors required by :func:`~MuyGPyS.gp.MuyGPS.posterior_mean` and
    :func:`~MuyGPyS.gp.MuyGPS.posterior_variance`.

    Args:
        metric:
            The metric to be used to compute distances.
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
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
    pairwise_diffs:
        A tensor of shape `(batch_count, nn_count, nn_count)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements.
    batch_nn_targets:
        Tensor of floats of shape `(batch_count, nn_count, response_count)`
        containing the expected response for each nearest neighbor of each batch
        element.
    """
    return _make_predict_tensors(
        metric,
        batch_indices,
        batch_nn_indices,
        test_features,
        train_features,
        train_targets,
    )


def make_train_tensors(
    metric: str,
    batch_indices: mm.ndarray,
    batch_nn_indices: mm.ndarray,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray, mm.ndarray, mm.ndarray]:
    """
    Create the distance and target tensors needed for training.

    Similar to :func:`~MuyGPyS.gp.data.make_predict_tensors` but returns the
    additional `batch_targets` matrix, which is only defined for a batch of
    training data.

    Args:
        metric:
            The metric to be used to compute distances.
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
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
    pairwise_diffs:
        A tensor of shape `(batch_count, nn_count, nn_count)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements.
    batch_targets:
        Matrix of floats of shape `(batch_count, response_count)` whose rows
        give the expected response for each batch element.
    batch_nn_targets:
        Tensor of floats of shape `(batch_count, nn_count, response_count)`
        containing the expected response for each nearest neighbor of each batch
        element.
    """
    return _make_train_tensors(
        metric, batch_indices, batch_nn_indices, train_features, train_targets
    )


def crosswise_tensors(
    data: mm.ndarray,
    nn_data: mm.ndarray,
    data_indices: mm.ndarray,
    nn_indices: mm.ndarray,
    metric: str = "l2",
) -> mm.ndarray:
    """
    Compute a matrix of distances between data and their nearest neighbors.

    Takes full datasets of records of interest `data` and neighbor candidates
    `nn_data` and produces the distances between each element of `data`
    indicated by `data_indices` and each of the nearest neighbors
    in `nn_data` as indicated by the corresponding rows of `nn_indices`. `data`
    and `nn_data` can refer to the same dataset.

    See the following example computing the crosswise distances between a batch
    of training data and their nearest neighbors.

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
        metric:
            The name of the metric to use in order to form distances. Supported
            values are `l2`, `F2`, `ip` (inner product, a distance only if
            data is normalized to the unit hypersphere), and `cosine`.

    Returns:
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
    """
    return _crosswise_tensors(
        data, nn_data, data_indices, nn_indices, metric=metric
    )


def pairwise_tensors(
    data: mm.ndarray,
    nn_indices: mm.ndarray,
    metric: str = "l2",
) -> mm.ndarray:
    """
    Compute a tensor of pairwise distances among sets of nearest neighbors.

    Takes a full dataset of records of interest `data` and produces the
    pairwise distances between the elements indicated by each row of
    `nn_indices`.

    Args:
        data:
            The data matrix of shape `(batch_count, feature_count)` containing
            batch elements.
        nn_indices:
            An integral matrix of shape (batch_count, nn_count) listing the
            nearest neighbor indices for the batch of data points.
        metric:
            The name of the metric to use in order to form distances. Supported
            values are `l2`, `F2`, `ip` (inner product, a distance only if
            data is normalized to the unit hypersphere), and `cosine`.

    Returns:
        A tensor of shape `(batch_count, nn_count, nn_count)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements.
    """
    return _pairwise_tensors(data, nn_indices, metric=metric)
