# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Tuple

import MuyGPyS._src.math.numpy as np


def _make_fast_predict_tensors(
    metric: str,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    num_train, _ = train_features.shape
    batch_nn_indices_fast = np.concatenate(
        (
            np.expand_dims(np.arange(0, num_train), axis=1),
            batch_nn_indices[:, :-1],
        ),
        axis=1,
    )

    pairwise_diffs_fast = _pairwise_tensor(
        train_features, batch_nn_indices_fast, metric=metric
    )
    batch_nn_targets_fast = train_targets[batch_nn_indices_fast]

    return pairwise_diffs_fast, batch_nn_targets_fast


def _make_predict_tensors(
    metric: str,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if test_features is None:
        test_features = train_features
    crosswise_diffs = _crosswise_tensor(
        test_features,
        train_features,
        batch_indices,
        batch_nn_indices,
        metric=metric,
    )
    pairwise_diffs = _pairwise_tensor(
        train_features, batch_nn_indices, metric=metric
    )
    batch_nn_targets = train_targets[batch_nn_indices, :]
    return crosswise_diffs, pairwise_diffs, batch_nn_targets


def _make_train_tensors(
    metric: str,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crosswise_diffs, pairwise_diffs, batch_nn_targets = _make_predict_tensors(
        metric,
        batch_indices,
        batch_nn_indices,
        train_features,
        train_features,
        train_targets,
    )
    batch_targets = train_targets[batch_indices, :]
    return crosswise_diffs, pairwise_diffs, batch_targets, batch_nn_targets


def _crosswise_tensor(
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


def _pairwise_tensor(
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


def _crosswise_diffs(locations: np.ndarray, points: np.ndarray) -> np.ndarray:
    return locations[:, None, :] - points


def _pairwise_diffs(points: np.ndarray) -> np.ndarray:
    if len(points.shape) == 3:
        return points[:, :, None, :] - points[:, None, :, :]
    elif len(points.shape) == 2:
        return points[:, None, :] - points[None, :, :]
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _F2(diffs: np.ndarray) -> np.ndarray:
    return np.sum(diffs**2, axis=-1)


def _l2(diffs: np.ndarray) -> np.ndarray:
    return np.sqrt(_F2(diffs))


def _fast_nn_update(
    nn_indices: np.ndarray,
) -> np.ndarray:
    train_count, _ = nn_indices.shape
    new_nn_indices = np.concatenate(
        (
            np.expand_dims(np.arange(0, train_count), axis=1),
            nn_indices[:, :-1],
        ),
        axis=1,
    )
    return new_nn_indices


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
