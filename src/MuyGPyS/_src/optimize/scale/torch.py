# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.torch as torch


def _analytic_scale_optim_unnormalized(
    Kin: torch.ndarray, nn_targets: torch.ndarray, **kwargs
) -> torch.ndarray:
    nn_targets = torch.atleast_3d(nn_targets)
    return torch.sum(
        torch.einsum(
            "ijk,ijk->ik", nn_targets, torch.linalg.solve(Kin, nn_targets)
        )
    )


def _analytic_scale_optim(
    Kin: torch.ndarray,
    nn_targets: torch.ndarray,
    batch_dim_count: int = 1,
    **kwargs,
) -> torch.ndarray:
    in_dim_count = (Kin.ndim - batch_dim_count) // 2

    batch_shape = Kin.shape[:batch_dim_count]
    in_shape = Kin.shape[batch_dim_count + in_dim_count :]

    batch_size = batch_shape.numel()
    in_size = in_shape.numel()

    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    nn_targets_flat = nn_targets.reshape(batch_shape + (in_size, 1))

    return _analytic_scale_optim_unnormalized(
        Kin_flat, nn_targets_flat, **kwargs
    ) / (batch_size * in_size)
