# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Tuple

import MuyGPyS._src.math.numpy as np


def _make_heteroscedastic_tensor(
    measurement_noise: np.ndarray,
    batch_nn_indices: np.ndarray,
) -> np.ndarray:
    return measurement_noise[batch_nn_indices]


def _make_fast_predict_tensors(
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    num_train = train_features.shape[0]
    batch_nn_indices_fast = np.concatenate(
        (
            np.expand_dims(np.arange(0, num_train), axis=1),
            batch_nn_indices[:, :-1],
        ),
        axis=1,
    )

    pairwise_diffs_fast = _pairwise_tensor(
        train_features, batch_nn_indices_fast
    )
    batch_nn_targets_fast = train_targets[batch_nn_indices_fast]

    return pairwise_diffs_fast, batch_nn_targets_fast


def _batch_features_tensor(
    features: np.ndarray,
    batch_indices: np.ndarray,
) -> np.ndarray:
    return features[batch_indices]


def _crosswise_tensor(
    data: np.ndarray,
    nn_data: np.ndarray,
    data_indices: np.ndarray,
    nn_indices: np.ndarray,
) -> np.ndarray:
    locations = data[data_indices]
    points = nn_data[nn_indices]
    if data.ndim == 1:
        return locations[..., :, None, None] - points[..., None]
    else:
        return locations[..., :, None, :] - points


def _pairwise_tensor(
    data: np.ndarray,
    nn_indices: np.ndarray,
) -> np.ndarray:
    points = data[nn_indices]
    if data.ndim == 1:
        return points[..., :, None, None] - points[..., None, :, None]
    else:
        return points[..., None, :] - points[..., None, :, :]


def _crosswise_differences(
    locations: np.ndarray, points: np.ndarray
) -> np.ndarray:
    return locations[:, None, :] - points


def _pairwise_differences(points: np.ndarray) -> np.ndarray:
    if points.ndim == 1:
        return np.subtract.outer(points, points)[:, :, None]
    elif points.ndim == 2:
        return points[:, None, :] - points[None, :, :]
    elif points.ndim == 3:
        return points[:, :, None, :] - points[:, None, :, :]
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _F2(diffs: np.ndarray) -> np.ndarray:
    return np.sum(diffs**2, axis=-1)


def _l2(diffs: np.ndarray) -> np.ndarray:
    return np.sqrt(_F2(diffs))


def _fast_nn_update(
    train_nn_indices: np.ndarray,
) -> np.ndarray:
    train_count = train_nn_indices.shape[0]
    new_nn_indices = np.concatenate(
        (
            np.expand_dims(np.arange(0, train_count), axis=1),
            train_nn_indices[:, :-1],
        ),
        axis=1,
    )
    return new_nn_indices
