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
from MuyGPyS.gp.kernels import SigmaSq, sigma_sq_scale
from MuyGPyS.gp.noise import HomoscedasticNoise, noise_perturb


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
            self.posterior_variance_fn = sigma_sq_scale(self._fn)

    def __call__(
        self,
        K: mm.ndarray,
        Kcross: mm.ndarray,
    ) -> mm.ndarray:
        return self._fn(K, Kcross, eps=self.eps(), sigma_sq=self.sigma_sq())

    def get_opt_fn(self) -> Callable:
        if isinstance(self.eps, HomoscedasticNoise):
            return self._get_opt_fn(
                self.posterior_variance_fn, self.eps, self.sigma_sq
            )
        else:
            raise TypeError(
                f"Noise parameter type {type(self.eps)} is not supported for "
                f"optimization!"
            )

    @staticmethod
    def _get_opt_fn(
        var_fn: Callable, eps: HomoscedasticNoise, sigma_sq: SigmaSq
    ) -> Callable:
        if not eps.fixed():

            def caller_fn(K, Kcross, **kwargs):
                return var_fn(K, Kcross, eps=kwargs["eps"], sigma_sq=sigma_sq())

        else:

            def caller_fn(K, Kcross, **kwargs):
                return var_fn(K, Kcross, eps=eps(), sigma_sq=sigma_sq())

        return caller_fn
