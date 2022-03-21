# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
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
    >>> from MuyGPyS.gp.distance import crosswise_distances
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
    >>> pairwise_dists = pairwise_distances(
    ...         train_features, batch_nn_inidices, metric="l2"
    ... )
    >>> crosswise_dists = crosswise_distances(
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
    >>> from MuyGPyS.gp.distance import crosswise_distances
    >>> train_features = load_train_features()
    >>> test_features = load_test_features()
    >>> nn_count = 10
    >>> nbrs_lookup = NN_Wrapper(
    ...         train_features, nn_count, nn_method="exact", algorithm="ball_tree"
    ... )
    >>> nn_indices, nn_dists = nbrs_lookup.get_nns(test_features)
    >>> test_count, _ = test_features.shape
    >>> indices = np.arange(test_count)
    >>> nn_indices, _ = nbrs_lookup.get_nns(test_features)
    >>> pairwise_dists = pairwise_distances(
    ...         train_features, nn_inidices, metric="l2"
    ... )
    >>> crosswise_dists = crosswise_distances(
    ...         test_features,
    ...         train_features,
    ...         indices,
    ...         nn_indices,
    ...         metric="l2"
    ... )

The helper functions :func:`MuyGPyS.gp.distance.make_regress_tensors` and
:func:`MuyGPyS.gp.distance.make_train_tensors` wrap these distances tensors and
also return the nearest neighbors sets' training targets and (in the latter
case) the training targets of the training batch. These functions are convenient
as the distance and target tensors are usually needed together.
"""


import numpy as np

from typing import Optional, Tuple

from MuyGPyS import config

if config.muygpys_jax_enabled is False:  # type: ignore
    from MuyGPyS._src.gp.numpy_distance import (
        _make_regress_tensors,
        _make_train_tensors,
        _crosswise_distances,
        _pairwise_distances,
    )
else:
    from MuyGPyS._src.gp.jax_distance import (
        _make_regress_tensors,
        _make_train_tensors,
        _crosswise_distances,
        _pairwise_distances,
    )


def make_regress_tensors(
    metric: str,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    test_features: Optional[np.ndarray],
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the distance and target tensors for regression.

    Creates the `crosswise_dists`, `pairwise_dists` and `batch_nn_targets`
    tensors required by :func:`MuyGPyS.gp.MuyGPyS.regress`.

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
    crosswise_dists:
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
    pairwise_dists:
        A tensor of shape `(batch_count, nn_count, nn_count,)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements.
    batch_nn_targets:
        Tensor of floats of shape `(batch_count, nn_count, response_count)`
        containing the expected response for each nearest neighbor of each batch
        element.
    """
    return _make_regress_tensors(
        metric,
        batch_indices,
        batch_nn_indices,
        test_features,
        train_features,
        train_targets,
    )


def make_train_tensors(
    metric: str,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the distance and target tensors needed for training.

    Similar to :func:`~MuyGPyS.gp.data.make_regress_tensors` but returns the
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
    crosswise_dists:
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
    pairwise_dists:
        A tensor of shape `(batch_count, nn_count, nn_count,)` whose latter two
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


def crosswise_distances(
    data: np.ndarray,
    nn_data: np.ndarray,
    data_indices: np.ndarray,
    nn_indices: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
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
    return _crosswise_distances(
        data, nn_data, data_indices, nn_indices, metric=metric
    )


def pairwise_distances(
    data: np.ndarray,
    nn_indices: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
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
        A tensor of shape `(batch_count, nn_count, nn_count,)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements.
    """
    return _pairwise_distances(data, nn_indices, metric=metric)
