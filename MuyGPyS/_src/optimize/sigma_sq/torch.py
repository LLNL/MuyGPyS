# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch


def _analytic_sigma_sq_optim_unnormalized(
    K: torch.Tensor,
    nn_targets: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(
        torch.einsum(
            "ijk,ijk->ik", nn_targets, torch.linalg.solve(K, nn_targets)
        ),
        dim=0,
    )


def _analytic_sigma_sq_optim(
    K: torch.Tensor,
    nn_targets: torch.Tensor,
) -> torch.Tensor:
    batch_count, nn_count, _ = nn_targets.shape
    return _analytic_sigma_sq_optim_unnormalized(K, nn_targets) / (
        nn_count * batch_count
    )
