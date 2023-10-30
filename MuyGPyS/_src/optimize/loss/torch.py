# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.torch as torch


def _cross_entropy_fn(
    predictions: torch.ndarray,
    targets: torch.ndarray,
) -> torch.ndarray:
    one_hot_targets = torch.where(targets > 0.0, 1.0, 0.0)
    softmax_predictions = predictions.softmax(dim=1)

    return (
        -torch.mean(
            one_hot_targets * torch.log(softmax_predictions)
            + (1 - one_hot_targets) * torch.log(1 - softmax_predictions)
        )
        * one_hot_targets.shape[0]
    )


def _mse_fn_unnormalized(
    predictions: torch.ndarray,
    targets: torch.ndarray,
) -> float:
    return torch.sum((predictions - targets) ** 2)


def _mse_fn(
    predictions: torch.ndarray,
    targets: torch.ndarray,
) -> float:
    batch_count, response_count = predictions.shape
    return _mse_fn_unnormalized(predictions, targets) / (
        batch_count * response_count
    )


def _lool_fn(
    predictions: torch.ndarray,
    targets: torch.ndarray,
    variances: torch.ndarray,
    scale: torch.ndarray,
) -> float:
    return _lool_fn_unscaled(
        predictions, targets, torch.unsqueeze(variances * scale, dim=1)
    )


def _lool_fn_unscaled(
    predictions: torch.ndarray, targets: torch.ndarray, variances: torch.ndarray
) -> float:
    return torch.sum(
        torch.divide((predictions - targets) ** 2, variances)
        + torch.log(variances)
    )


def _pseudo_huber_fn(
    predictions: torch.ndarray,
    targets: torch.ndarray,
    boundary_scale: float = 1.5,
) -> float:
    return boundary_scale**2 * torch.sum(
        torch.sqrt(1 + torch.divide(targets - predictions, boundary_scale) ** 2)
        - 1
    )


def _looph_fn_unscaled(
    predictions: torch.ndarray,
    targets: torch.ndarray,
    variances: torch.ndarray,
    boundary_scale: float = 3.0,
) -> float:
    boundary_scale_sq = boundary_scale**2
    return torch.sum(
        2
        * boundary_scale_sq
        * (
            torch.sqrt(
                1
                + torch.divide(
                    (targets - predictions) ** 2, boundary_scale_sq * variances
                )
            )
            - 1
        )
        + torch.log(variances)
    )


def _looph_fn(
    predictions: torch.ndarray,
    targets: torch.ndarray,
    variances: torch.ndarray,
    scale: torch.ndarray,
    boundary_scale: float = 3.0,
) -> float:
    return _looph_fn_unscaled(
        predictions,
        targets,
        torch.outer(variances, scale),
        boundary_scale=boundary_scale,
    )
