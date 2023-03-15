# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import _muygps_diagonal_variance
from MuyGPyS._src.gp.noise import _homoscedastic_perturb
from MuyGPyS.gp.kernels import SigmaSq, sigma_sq_scale, sigma_sq_apply
from MuyGPyS.gp.noise import HomoscedasticNoise, noise_perturb, noise_apply


class PosteriorVariance:
    def __init__(
        self,
        eps: HomoscedasticNoise,
        sigma_sq: SigmaSq,
        apply_sigma_sq=True,
        **kwargs,
    ):
        self.eps = eps
        self.sigma_sq = sigma_sq
        self._fn = _muygps_diagonal_variance
        if isinstance(self.eps, HomoscedasticNoise):
            self._fn = noise_perturb(_homoscedastic_perturb)(self._fn)
        else:
            raise ValueError(f"Noise model {type(self.eps)} is not supported")
        if apply_sigma_sq is True:
            self._fn = sigma_sq_scale(self._fn)

    def __call__(
        self,
        K: mm.ndarray,
        Kcross: mm.ndarray,
    ) -> mm.ndarray:
        return self._fn(K, Kcross, eps=self.eps(), sigma_sq=self.sigma_sq())

    def get_opt_fn(self) -> Callable:
        return self._get_opt_fn(self._fn, self.eps, self.sigma_sq)

    @staticmethod
    def _get_opt_fn(
        var_fn: Callable, eps: HomoscedasticNoise, sigma_sq: SigmaSq
    ) -> Callable:
        if isinstance(eps, HomoscedasticNoise):
            opt_fn = noise_apply(var_fn, eps)
        else:
            raise TypeError(
                f"Noise parameter type {type(eps)} is not supported for "
                f"optimization!"
            )

        opt_fn = sigma_sq_apply(opt_fn, sigma_sq)
        return opt_fn
