# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch


def _muygps_compute_solve(
    K: torch.Tensor,
    Kcross: torch.Tensor,
    batch_nn_targets: torch.Tensor,
) -> torch.Tensor:
    batch_count, nn_count, response_count = batch_nn_targets.shape
    responses = Kcross.reshape(batch_count, 1, nn_count) @ torch.linalg.solve(
        K, batch_nn_targets
    )
    return responses.reshape(batch_count, response_count)


def _muygps_compute_diagonal_variance(
    K: torch.Tensor,
    Kcross: torch.Tensor,
) -> torch.Tensor:
    batch_count, nn_count = Kcross.shape
    return 1 - torch.sum(
        Kcross
        * torch.linalg.solve(
            K, Kcross.reshape(batch_count, nn_count, 1)
        ).reshape(batch_count, nn_count),
        dim=1,
    )


def _muygps_fast_regress_solve(
    Kcross: torch.Tensor,
    coeffs_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("ij,ijk->ik", Kcross, coeffs_tensor)


def _mmuygps_fast_regress_solve(
    Kcross: torch.Tensor,
    coeffs_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("ijk,ijk->ik", Kcross, coeffs_tensor)


def _muygps_fast_regress_precompute(
    K: torch.Tensor,
    eps: float,
    train_nn_targets_fast: torch.Tensor,
) -> torch.Tensor:
    _, nn_count, _ = K.shape
    coeffs_tensor = torch.linalg.solve(
        K + eps * torch.eye(nn_count), train_nn_targets_fast
    )
    return coeffs_tensor
