# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.torch as torch


def _analytic_scale_optim_unnormalized(
    Kin: torch.ndarray,
    nn_targets: torch.ndarray,
) -> torch.ndarray:
    return torch.sum(
        torch.einsum(
            "ijk,ijk->ik", nn_targets, torch.linalg.solve(Kin, nn_targets)
        ),
        axis=0,
    )


def _analytic_scale_optim(
    Kin: torch.ndarray,
    nn_targets: torch.ndarray,
) -> torch.ndarray:
    batch_count, nn_count, _ = nn_targets.shape
    return _analytic_scale_optim_unnormalized(Kin, nn_targets) / (
        nn_count * batch_count
    )
