# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Distance functions

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
    >>> indices = jnp.arange(test_count)
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


import jax.numpy as jnp

from typing import Tuple

from functools import partial
from jax import jit
from sklearn.metrics.pairwise import cosine_similarity


@partial(jit, static_argnums=(0,))
def make_regress_tensors(
    metric: str,
    batch_indices: jnp.ndarray,
    batch_nn_indices: jnp.ndarray,
    test_features: jnp.ndarray,
    train_features: jnp.ndarray,
    train_targets: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    if test_features is None:
        test_features = train_features
    crosswise_dists = crosswise_distances(
        test_features,
        train_features,
        batch_indices,
        batch_nn_indices,
        metric=metric,
    )
    pairwise_dists = pairwise_distances(
        train_features, batch_nn_indices, metric=metric
    )
    batch_nn_targets = train_targets[batch_nn_indices, :]
    return crosswise_dists, pairwise_dists, batch_nn_targets


@partial(jit, static_argnums=(0,))
def make_train_tensors(
    metric: str,
    batch_indices: jnp.ndarray,
    batch_nn_indices: jnp.ndarray,
    train_features: jnp.ndarray,
    train_targets: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    crosswise_dists, pairwise_dists, batch_nn_targets = make_regress_tensors(
        metric,
        batch_indices,
        batch_nn_indices,
        train_features,
        train_features,
        train_targets,
    )
    batch_targets = train_targets[batch_indices, :]
    return crosswise_dists, pairwise_dists, batch_targets, batch_nn_targets


@partial(jit, static_argnums=(4,))
def crosswise_distances(
    data: jnp.ndarray,
    nn_data: jnp.ndarray,
    data_indices: jnp.ndarray,
    nn_indices: jnp.ndarray,
    metric: str = "l2",
) -> jnp.ndarray:
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
    locations = data[data_indices]
    points = nn_data[nn_indices]
    if metric == "l2":
        diffs = _crosswise_diffs(locations, points)
        return _l2(diffs)
    elif metric == "F2":
        diffs = _crosswise_diffs(locations, points)
        return _F2(diffs)
    # elif metric == "ip":
    #     return _crosswise_prods(locations, points)
    # elif metric == "cosine":
    #     return _crosswise_cosine(locations, points)
    else:
        raise ValueError(f"Metric {metric} is not supported!")


@jit
def _crosswise_diffs(
    locations: jnp.ndarray, points: jnp.ndarray
) -> jnp.ndarray:
    return locations[:, None, :] - points


@partial(jit, static_argnums=(2,))
def pairwise_distances(
    data: jnp.ndarray,
    nn_indices: jnp.ndarray,
    metric: str = "l2",
) -> jnp.ndarray:
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
    points = data[nn_indices]
    if metric == "l2":
        diffs = _diffs(points)
        return _l2(diffs)
    elif metric == "F2":
        diffs = _diffs(points)
        return _F2(diffs)
    # elif metric == "ip":
    #     return _prods(points)
    # elif metric == "cosine":
    #     return _cosine(points)
    else:
        raise ValueError(f"Metric {metric} is not supported!")


@jit
def _diffs(points: jnp.ndarray) -> jnp.ndarray:
    return points[:, :, None, :] - points[:, None, :, :]


@jit
def _F2(diffs: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(diffs ** 2, axis=-1)


@jit
def _l2(diffs: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(_F2(diffs))