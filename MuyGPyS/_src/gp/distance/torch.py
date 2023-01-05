import torch

from typing import Tuple


def _make_fast_regress_tensors(
    metric: str,
    batch_nn_indices: torch.Tensor,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_train, _ = train_features.shape
    batch_nn_indices_fast = torch.cat(
        (
            torch.unsqueeze(torch.arange(0, num_train), dim=1),
            batch_nn_indices[:, :-1],
        ),
        dim=1,
    )

    pairwise_dists_fast = _pairwise_distances(
        train_features, batch_nn_indices_fast, metric=metric
    )
    batch_nn_targets_fast = train_targets[batch_nn_indices_fast]

    return pairwise_dists_fast, batch_nn_targets_fast


def _make_regress_tensors(
    metric: str,
    batch_indices: torch.Tensor,
    batch_nn_indices: torch.Tensor,
    test_features: torch.Tensor,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    batch_indices: torch.Tensor,
    batch_nn_indices: torch.Tensor,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    data: torch.Tensor,
    nn_data: torch.Tensor,
    data_indices: torch.Tensor,
    nn_indices: torch.Tensor,
    metric: str = "l2",
) -> torch.Tensor:
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
    data: torch.Tensor,
    nn_indices: torch.Tensor,
    metric: str = "l2",
) -> torch.Tensor:
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
    locations: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    return locations[:, None, :] - points


def _pairwise_diffs(points: torch.Tensor) -> torch.Tensor:
    if len(points.shape) == 3:
        return points[:, :, None, :] - points[:, None, :, :]
    elif len(points.shape) == 2:
        return points[:, None, :] - points[None, :, :]
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


def _F2(diffs: torch.Tensor) -> torch.Tensor:
    return torch.sum(diffs**2, dim=-1)


def _l2(diffs: torch.Tensor) -> torch.Tensor:
    return torch.norm(diffs, dim=-1)


def _fast_nn_update(
    nn_indices: torch.Tensor,
) -> torch.Tensor:
    train_count, _ = nn_indices.shape
    new_nn_indices = torch.cat(
        (
            torch.unsqueeze(torch.arange(0, train_count), dim=1),
            nn_indices[:, :-1],
        ),
        dim=1,
    )
    return new_nn_indices
