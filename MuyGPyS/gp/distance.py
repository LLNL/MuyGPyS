# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def pairwise_distances(data, nn_indices, metric="l2"):
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
        return 1 - points @ point.T
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
