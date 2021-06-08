# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def crosswise_distances(data, nn_data, data_indices, nn_indices, metric="l2"):
    """
    Form a distance tensor from nearest neighbor indices into a dataset.

    Parameters
    ----------
    data : numpy.ndarray(float), shape = ``(data_count, feature_count)''
        The data matrix containing batch elements.
    nn_data : numpy.ndarray(float), shape = ``(data_count, feature_count)''
        The data matrix containing the universe of candidate neighbors for
        the batch elements. Might be the same as ``data''.
    indices : np.ndarray(int), shape = ``(batch_count,)''
        The integer indices of the batch.
    nn_indices : numpy.ndarray(int), shape = ``(batch_size, nn_count)''
        A matrix listing the nearest neighbor indices for a batch of data
        points.
    metric : string
        The name of the metric to use in order to form distances.
        NOTE[bwp] supported values are ``l2'', ``F2'', ``ip'' (inner product,
        a distance only if data is normalized to the unit hypersphere), and
        ``cosine''.

    Returns
    -------
    numpy.ndarray(float), shape = ``(batch_count, nn_count)''
        A matrix whose rows list the distance of the corresponding batch
        element to each of its nearest neighbors.
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


def _crosswise_diffs(locations, points):
    return np.array(
        [
            [locations[i, :] - points[i, j, :] for j in range(points.shape[1])]
            for i in range(points.shape[0])
        ]
    )


def _crosswise_F2(locations, points):
    return np.array(
        [
            [
                _F2(locations[i, :] - points[i, j, :])
                for j in range(points.shape[1])
            ]
            for i in range(points.shape[0])
        ]
    )


def _crosswise_l2(locations, points):
    return np.array(
        [
            [
                _l2(locations[i, :] - points[i, j, :])
                for j in range(points.shape[1])
            ]
            for i in range(points.shape[0])
        ]
    )


def _crosswise_prods(locations, points):
    return 1 - np.array(
        [
            [locations[i, :] @ points[i, j, :] for j in range(points.shape[1])]
            for i in range(points.shape[0])
        ]
    )


def _crosswise_cosine(locations, points):
    return 1 - np.array(
        [
            [
                cosine_similarity(locations[i, :], points[i, j, :])
                for j in range(points.shape[1])
            ]
            for i in range(points.shape[0])
        ]
    )


def pairwise_distances(data, nn_indices, metric="l2"):
    """
    Form a distance tensor from nearest neighbor indices into a dataset.

    Parameters
    ----------
    data : numpy.ndarray(float), shape = ``(data_count, feature_count)''
        The full data matrix.
    nn_indices : numpy.ndarray(int), shape = ``(batch_size, nn_count)''
        A matrix listing the nearest neighbor indices for a batch of data
        points, not necessarily contained in the data.
    metric : string
        The name of the metric to use in order to form distances.
        NOTE[bwp] supported values are ``l2'', ``F2'', ``ip'' (inner product,
        a distance only if data is normalized to the unit hypersphere), and
        ``cosine''.

    Returns
    -------
    numpy.ndarray(float), shape = ``(batch_count, nn_count, nn_count,)''
        A tensor whose latter two dimensions contain square matrices containing
        the pairwise distances between the nearest neighbors of the batch
        elements.
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


def _diffs(points):
    if len(points.shape) == 3:
        return points[:, :, None, :] - points[:, None, :, :]
    elif len(points.shape) == 2:
        return points[:, None, :] - points[None, :, :]
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _F2(diffs):
    return np.sum(diffs ** 2, axis=-1)


def _l2(diffs):
    return np.sqrt(_F2(diffs))


def _prods(points):
    if len(points.shape) == 3:
        return 1 - np.array([mat @ mat.T for mat in points])
    elif len(points.shape) == 2:
        return 1 - points @ points.T
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _cosine(points):
    if len(points.shape) == 3:
        return 1 - np.array([cosine_similarity(mat) for mat in points])
    elif len(points.shape) == 2:
        return 1 - cosine_similarity(points)
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


# def cosine(x_locs, z_locs=None):
#     x_diags = np.sum(x_locs ** 2, axis=1)
#     if z_locs is None:
#         z_locs = x_locs
#         z_diags = x_diags
#     else:
#         z_diags = np.sum(z_locs ** 2, axis=1)
#         if x_locs.shape[1] != z_locs.shape[1]:
#             raise ValueError(
#                 f"x_locs shape {x_locs.shape} is incompatible with z_locs shape"
#                 f" {z_locs.shape}"
#             )
#     cross_diff_tensor = 1 - np.einsum(
#         "bj, bij -> bi",
#         x_locs[batch_indices],
#         z_locs[batch_nn_indices],
#     )
#     cross_diag_tensor = np.einsum(
#         "b, bi -> bi",
#         np.sqrt(train_diags[batch_indices]),
#         np.sqrt(train_diags[batch_nn_indices]),
#     )
#     cross_dist_tensor = cross_diff_tensor / cross_diag_tensor
#     return dist_tensor
