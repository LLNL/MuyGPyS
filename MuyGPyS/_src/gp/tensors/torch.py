# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Tuple

import MuyGPyS._src.math.torch as torch


def _make_fast_predict_tensors(
    metric: str,
    batch_nn_indices: torch.ndarray,
    train_features: torch.ndarray,
    train_targets: torch.ndarray,
) -> Tuple[torch.ndarray, torch.ndarray]:
    num_train, _ = train_features.shape
    batch_nn_indices_fast = torch.cat(
        (
            torch.unsqueeze(torch.arange(0, num_train), dim=1),
            batch_nn_indices[:, :-1],
        ),
        dim=1,
    )

    pairwise_dists_fast = _pairwise_tensor(
        train_features, batch_nn_indices_fast, metric=metric
    )
    batch_nn_targets_fast = train_targets[batch_nn_indices_fast]

    return pairwise_dists_fast, batch_nn_targets_fast


def _make_predict_tensors(
    metric: str,
    batch_indices: torch.ndarray,
    batch_nn_indices: torch.ndarray,
    test_features: torch.ndarray,
    train_features: torch.ndarray,
    train_targets: torch.ndarray,
) -> Tuple[torch.ndarray, torch.ndarray, torch.ndarray]:
    if test_features is None:
        test_features = train_features
    crosswise_dists = _crosswise_tensor(
        test_features,
        train_features,
        batch_indices,
        batch_nn_indices,
        metric=metric,
    )
    pairwise_dists = _pairwise_tensor(
        train_features, batch_nn_indices, metric=metric
    )
    batch_nn_targets = train_targets[batch_nn_indices, :]
    return crosswise_dists, pairwise_dists, batch_nn_targets


def _make_train_tensors(
    metric: str,
    batch_indices: torch.ndarray,
    batch_nn_indices: torch.ndarray,
    train_features: torch.ndarray,
    train_targets: torch.ndarray,
) -> Tuple[torch.ndarray, torch.ndarray, torch.ndarray, torch.ndarray]:
    crosswise_dists, pairwise_dists, batch_nn_targets = _make_predict_tensors(
        metric,
        batch_indices,
        batch_nn_indices,
        train_features,
        train_features,
        train_targets,
    )
    batch_targets = train_targets[batch_indices, :]
    return crosswise_dists, pairwise_dists, batch_targets, batch_nn_targets


def _crosswise_tensor(
    data: torch.ndarray,
    nn_data: torch.ndarray,
    data_indices: torch.ndarray,
    nn_indices: torch.ndarray,
    metric: str = "l2",
) -> torch.ndarray:
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
    data: torch.ndarray,
    nn_indices: torch.ndarray,
    metric: str = "l2",
) -> torch.ndarray:
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


def _crosswise_diffs(
    locations: torch.ndarray, points: torch.ndarray
) -> torch.ndarray:
    return locations[:, None, :] - points


def _pairwise_diffs(points: torch.ndarray) -> torch.ndarray:
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
    nn_indices: torch.ndarray,
) -> torch.ndarray:
    train_count, _ = nn_indices.shape
    new_nn_indices = torch.cat(
        (
            torch.unsqueeze(torch.arange(0, train_count), dim=1),
            nn_indices[:, :-1],
        ),
        dim=1,
    )
    return new_nn_indices