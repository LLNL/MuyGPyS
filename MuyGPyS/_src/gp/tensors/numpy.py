# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
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
    num_train, _ = train_features.shape
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


def _make_predict_tensors(
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if test_features is None:
        test_features = train_features
    crosswise_diffs = _crosswise_tensor(
        test_features, train_features, batch_indices, batch_nn_indices
    )
    pairwise_diffs = _pairwise_tensor(train_features, batch_nn_indices)
    batch_nn_targets = train_targets[batch_nn_indices, :]
    return crosswise_diffs, pairwise_diffs, batch_nn_targets


def _make_train_tensors(
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crosswise_diffs, pairwise_diffs, batch_nn_targets = _make_predict_tensors(
        batch_indices,
        batch_nn_indices,
        train_features,
        train_features,
        train_targets,
    )
    batch_targets = train_targets[batch_indices, :]
    return crosswise_diffs, pairwise_diffs, batch_targets, batch_nn_targets


def _batch_features_tensor(
    features: np.ndarray,
    batch_indices: np.ndarray,
) -> np.ndarray:
    return features[batch_indices, :]


def _crosswise_tensor(
    data: np.ndarray,
    nn_data: np.ndarray,
    data_indices: np.ndarray,
    nn_indices: np.ndarray,
) -> np.ndarray:
    locations = data[data_indices]
    points = nn_data[nn_indices]
    return _crosswise_differences(locations, points)


def _pairwise_tensor(
    data: np.ndarray,
    nn_indices: np.ndarray,
) -> np.ndarray:
    points = data[nn_indices]
    return _pairwise_differences(points)


def _crosswise_differences(
    locations: np.ndarray, points: np.ndarray
) -> np.ndarray:
    return locations[:, None, :] - points


def _pairwise_differences(points: np.ndarray) -> np.ndarray:
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
    train_nn_indices: np.ndarray,
) -> np.ndarray:
    train_count, _ = train_nn_indices.shape
    new_nn_indices = np.concatenate(
        (
            np.expand_dims(np.arange(0, train_count), axis=1),
            train_nn_indices[:, :-1],
        ),
        axis=1,
    )
    return new_nn_indices
