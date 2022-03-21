# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from typing import Tuple

# from sklearn.metrics.pairwise import cosine_similarity


def _make_regress_tensors(
    metric: str,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if test_features is None:
        test_features = train_features
    crosswise_dists = _crosswise_distances(
        test_features,
        train_features,
        batch_indices,
        batch_nn_indices,
        metric=metric,
    )
    pairwise_dists = _pairwise_distances(
        train_features, batch_nn_indices, metric=metric
    )
    batch_nn_targets = train_targets[batch_nn_indices, :]
    return crosswise_dists, pairwise_dists, batch_nn_targets


def _make_train_tensors(
    metric: str,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crosswise_dists, pairwise_dists, batch_nn_targets = _make_regress_tensors(
        metric,
        batch_indices,
        batch_nn_indices,
        train_features,
        train_features,
        train_targets,
    )
    batch_targets = train_targets[batch_indices, :]
    return crosswise_dists, pairwise_dists, batch_targets, batch_nn_targets


def _crosswise_distances(
    data: np.ndarray,
    nn_data: np.ndarray,
    data_indices: np.ndarray,
    nn_indices: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
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


def _pairwise_distances(
    data: np.ndarray,
    nn_indices: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
    points = data[nn_indices]
    if metric == "l2":
        diffs = _pairwise_diffs(points)
        return _l2(diffs)
    elif metric == "F2":
        diffs = _pairwise_diffs(points)
        return _F2(diffs)
    # elif metric == "ip":
    #     return _pairwise_prods(points)
    # elif metric == "cosine":
    #     return _pairwise_cosine(points)
    else:
        raise ValueError(f"Metric {metric} is not supported!")


def _crosswise_diffs(locations: np.array, points: np.array) -> np.array:
    return locations[:, None, :] - points


def _pairwise_diffs(points: np.array) -> np.array:
    if len(points.shape) == 3:
        return points[:, :, None, :] - points[:, None, :, :]
    elif len(points.shape) == 2:
        return points[:, None, :] - points[None, :, :]
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _F2(diffs: np.array) -> np.array:
    return np.sum(diffs**2, axis=-1)


def _l2(diffs: np.array) -> np.array:
    return np.sqrt(_F2(diffs))


# def _prods(points: np.array) -> np.array:
#     if len(points.shape) == 3:
#         return 1 - np.array([mat @ mat.T for mat in points])
#     elif len(points.shape) == 2:
#         return 1 - points @ points.T
#     else:
#         raise ValueError(f"points shape {points.shape} is not supported.")


# def _cosine(points: np.array) -> np.array:
#     if len(points.shape) == 3:
#         return 1 - np.array([cosine_similarity(mat) for mat in points])
#     elif len(points.shape) == 2:
#         return 1 - cosine_similarity(points)
#     else:
#         raise ValueError(f"points shape {points.shape} is not supported.")
