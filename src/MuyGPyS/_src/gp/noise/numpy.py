# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _homoscedastic_perturb(
    Kin: np.ndarray, noise_variance: float
) -> np.ndarray:
    if Kin.ndim == 3:
        _, nn_count, _ = Kin.shape
        return Kin + noise_variance * np.eye(nn_count)
    elif Kin.ndim == 5:
        b, in_count, nn_count, in_count2, nn_count2 = Kin.shape
        assert nn_count == nn_count2
        assert in_count == in_count2
        all_count = in_count * nn_count
        Kin_flat = Kin.reshape(b, all_count, all_count)
        Kin_flat = Kin_flat + noise_variance * np.eye(all_count)
        return Kin_flat.reshape(b, in_count, nn_count, in_count, nn_count)
    else:
        raise ValueError(
            "homoscedastic perturbation is not implemented for tensors of "
            f"shape {Kin.shape}"
        )


def _shear_perturb33(Kin: np.ndarray, noise_variance: float) -> np.ndarray:
    convergence_variance = noise_variance * 2
    if Kin.ndim == 5:
        b, in_count, nn_count, in_count2, nn_count2 = Kin.shape
        assert nn_count == nn_count2
        assert in_count == in_count2
        assert in_count == 3
        all_count = in_count * nn_count
        Kin_flat = Kin.reshape(b, all_count, all_count)
        nugget = np.diag(
            np.hstack(
                (
                    convergence_variance * np.ones(nn_count),
                    noise_variance * np.ones(2 * nn_count),
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
    Kin: np.ndarray, noise_variances: np.ndarray
) -> np.ndarray:
    ret = Kin.copy()
    batch_count, nn_count, _ = Kin.shape
    indices = (
        np.repeat(range(batch_count), nn_count),
        np.tile(np.arange(nn_count), batch_count),
        np.tile(np.arange(nn_count), batch_count),
    )
    ret[indices] += noise_variances.flatten()
    return ret
