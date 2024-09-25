# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.torch as torch


def _homoscedastic_perturb(
    Kin: torch.ndarray, noise_variance: float
) -> torch.ndarray:
    if Kin.ndim == 3:
        _, nn_count, _ = Kin.shape
        return Kin + noise_variance * torch.eye(nn_count)
    elif Kin.ndim == 5:
        b, in_count, nn_count, in_count2, nn_count2 = Kin.shape
        assert nn_count == nn_count2
        assert in_count == in_count2
        all_count = in_count * nn_count
        Kin_flat = Kin.reshape(b, all_count, all_count)
        Kin_flat = Kin_flat + noise_variance * torch.eye(all_count)
        return Kin_flat.reshape(b, in_count, nn_count, in_count, nn_count)
    else:
        raise ValueError(
            "homoscedastic perturbation is not implemented for tensors of "
            f"shape {Kin.shape}"
        )


def _shear_perturb33(
    Kin: torch.ndarray, noise_variance: float
) -> torch.ndarray:
    convergence_variance = noise_variance * 2
    if Kin.ndim == 5:
        b, in_count, nn_count, in_count2, nn_count2 = Kin.shape
        assert nn_count == nn_count2
        assert in_count == in_count2
        assert in_count == 3
        all_count = in_count * nn_count
        Kin_flat = Kin.reshape(b, all_count, all_count)
        nugget = torch.diag(
            torch.hstack(
                (
                    convergence_variance * torch.ones(nn_count),
                    noise_variance * torch.ones(2 * nn_count),
                )
            )
        )
        Kin_flat = Kin_flat + nugget
        return Kin_flat.reshape(b, in_count, nn_count, in_count, nn_count)
    else:
        raise ValueError(
            "homoscedastic perturbation is not implemented for tensors of "
            f"shape {Kin.shape}"
        )


def _heteroscedastic_perturb(
    Kin: torch.ndarray, noise_variances: torch.ndarray
) -> torch.ndarray:
    ret = Kin.clone()
    batch_count, nn_count, _ = Kin.shape
    indices = (
        torch.repeat(torch.arange(batch_count), nn_count),
        torch.arange(nn_count).repeat(batch_count),
        torch.arange(nn_count).repeat(batch_count),
    )
    ret[indices] += noise_variances.flatten()
    return ret
