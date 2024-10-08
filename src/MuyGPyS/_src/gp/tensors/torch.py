# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Tuple

import MuyGPyS._src.math.torch as torch


def _make_heteroscedastic_tensor(
    measurement_noise: torch.ndarray,
    batch_nn_indices: torch.ndarray,
) -> torch.ndarray:
    return measurement_noise[batch_nn_indices]


def _make_fast_predict_tensors(
    batch_nn_indices: torch.ndarray,
    train_features: torch.ndarray,
    train_targets: torch.ndarray,
) -> Tuple[torch.ndarray, torch.ndarray]:
    num_train = train_features.shape[0]
    batch_nn_indices_fast = torch.cat(
        (
            torch.unsqueeze(torch.arange(0, num_train), dim=1),
            batch_nn_indices[:, :-1],
        ),
        dim=1,
    )

    pairwise_dists_fast = _pairwise_tensor(
        train_features, batch_nn_indices_fast
    )
    batch_nn_targets_fast = train_targets[batch_nn_indices_fast]

    return pairwise_dists_fast, batch_nn_targets_fast


def _batch_features_tensor(
    features: torch.ndarray,
    batch_indices: torch.ndarray,
) -> torch.ndarray:
    return features[batch_indices]


def _crosswise_tensor(
    data: torch.ndarray,
    nn_data: torch.ndarray,
    data_indices: torch.ndarray,
    nn_indices: torch.ndarray,
) -> torch.ndarray:
    locations = data[data_indices]
    points = nn_data[nn_indices]
    return _crosswise_differences(locations, points)


def _pairwise_tensor(
    data: torch.ndarray,
    nn_indices: torch.ndarray,
) -> torch.ndarray:
    points = data[nn_indices]
    return _pairwise_differences(points)


def _crosswise_differences(
    locations: torch.ndarray, points: torch.ndarray
) -> torch.ndarray:
    return locations[:, None, :] - points


def _pairwise_differences(points: torch.ndarray) -> torch.ndarray:
    if len(points.shape) == 3:
        return points[:, :, None, :] - points[:, None, :, :]
    elif len(points.shape) == 2:
        return points[:, None, :] - points[None, :, :]
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _F2(diffs: torch.ndarray) -> torch.ndarray:
    return torch.sum(diffs**2, axis=-1)


def _l2(diffs: torch.ndarray) -> torch.ndarray:
    return torch.norm(diffs, dim=-1)


def _fast_nn_update(
    train_nn_indices: torch.ndarray,
) -> torch.ndarray:
    train_count = train_nn_indices.shape[0]
    new_nn_indices = torch.cat(
        (
            torch.unsqueeze(torch.arange(0, train_count), dim=1),
            train_nn_indices[:, :-1],
        ),
        dim=1,
    )
    return new_nn_indices
