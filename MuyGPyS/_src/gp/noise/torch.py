# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.torch as torch


def _homoscedastic_perturb(
    K: torch.ndarray, noise_variance: float
) -> torch.ndarray:
    _, nn_count, _ = K.shape
    return K + noise_variance * torch.eye(nn_count)


def _heteroscedastic_perturb(
    K: torch.ndarray, noise_variances: torch.ndarray
) -> torch.ndarray:
    ret = K.clone()
    batch_count, nn_count, _ = K.shape
    indices = (
        torch.repeat(torch.arange(batch_count), nn_count),
        torch.arange(nn_count).repeat(batch_count),
        torch.arange(nn_count).repeat(batch_count),
    )
    ret[indices] += noise_variances.flatten()
    return ret
