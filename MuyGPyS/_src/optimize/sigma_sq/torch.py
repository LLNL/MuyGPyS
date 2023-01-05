# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch


def _analytic_sigma_sq_optim_unnormalized(
    K: torch.Tensor,
    nn_targets: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    _, nn_count, _ = nn_targets.shape
    return torch.sum(
        torch.einsum(
            "ijk,ijk->ik",
            nn_targets,
            torch.linalg.solve(K + eps * torch.eye(nn_count), nn_targets),
        ),
        dim=0,
    )


def _analytic_sigma_sq_optim(
    K: torch.Tensor,
    nn_targets: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    batch_count, nn_count, _ = nn_targets.shape
    return _analytic_sigma_sq_optim_unnormalized(K, nn_targets, eps) / (
        nn_count * batch_count
    )
