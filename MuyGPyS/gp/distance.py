# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

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

"""


import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


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
            An integral matrix of shape (batch_size, nn_count) listing the
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
        return _crosswise_l2(locations, points)
    elif metric == "F2":
        return _crosswise_F2(locations, points)
    elif metric == "ip":
        return _crosswise_prods(locations, points)
    elif metric == "cosine":
        return _crosswise_cosine(points)
    else:
        raise ValueError(f"Metric {metric} is not supported!")


def _crosswise_diffs(locations: np.array, points: np.array) -> np.array:
    return np.array(
        [
            [locations[i, :] - points[i, j, :] for j in range(points.shape[1])]
            for i in range(points.shape[0])
        ]
    )


def _crosswise_F2(locations: np.array, points: np.array) -> np.array:
    return np.array(
        [
            [
                _F2(locations[i, :] - points[i, j, :])
                for j in range(points.shape[1])
            ]
            for i in range(points.shape[0])
        ]
    )


def _crosswise_l2(locations: np.array, points: np.array) -> np.array:
    return np.array(
        [
            [
                _l2(locations[i, :] - points[i, j, :])
                for j in range(points.shape[1])
            ]
            for i in range(points.shape[0])
        ]
    )


def _crosswise_prods(locations: np.array, points: np.array) -> np.array:
    return 1 - np.array(
        [
            [locations[i, :] @ points[i, j, :] for j in range(points.shape[1])]
            for i in range(points.shape[0])
        ]
    )


def _crosswise_cosine(locations: np.array, points: np.array) -> np.array:
    return 1 - np.array(
        [
            [
                cosine_similarity(locations[i, :], points[i, j, :])
                for j in range(points.shape[1])
            ]
            for i in range(points.shape[0])
        ]
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
            The data matrix of shape `(data_count, feature_count)` containing
            batch elements.
        nn_indices:
            An integral matrix of shape (batch_size, nn_count) listing the
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
    elif metric == "ip":
        return _prods(points)
    elif metric == "cosine":
        return _cosine(points)
    else:
        raise ValueError(f"Metric {metric} is not supported!")


def _diffs(points: np.array) -> np.array:
    if len(points.shape) == 3:
        return points[:, :, None, :] - points[:, None, :, :]
    elif len(points.shape) == 2:
        return points[:, None, :] - points[None, :, :]
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _F2(diffs: np.array) -> np.array:
    return np.sum(diffs ** 2, axis=-1)


def _l2(diffs: np.array) -> np.array:
    return np.sqrt(_F2(diffs))


def _prods(points: np.array) -> np.array:
    if len(points.shape) == 3:
        return 1 - np.array([mat @ mat.T for mat in points])
    elif len(points.shape) == 2:
        return 1 - points @ points.T
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _cosine(points: np.array) -> np.array:
    if len(points.shape) == 3:
        return 1 - np.array([cosine_similarity(mat) for mat in points])
    elif len(points.shape) == 2:
        return 1 - cosine_similarity(points)
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")
