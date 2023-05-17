# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable, Union

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import _muygps_diagonal_variance
from MuyGPyS.gp.sigma_sq import SigmaSq, sigma_sq_scale, sigma_sq_apply
from MuyGPyS.gp.noise import (
    HomoscedasticNoise,
    HeteroscedasticNoise,
    NullNoise,
    perturb_with_noise_model,
)


class PosteriorVariance:
    def __init__(
        self,
        eps: Union[HomoscedasticNoise, HeteroscedasticNoise, NullNoise],
        sigma_sq: SigmaSq,
        apply_sigma_sq=True,
        **kwargs,
    ):
        self.eps = eps
        self.sigma_sq = sigma_sq
        self._fn = _muygps_diagonal_variance
        self._fn = perturb_with_noise_model(self._fn, self.eps)
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
        var_fn: Callable,
        eps: Union[HomoscedasticNoise, HeteroscedasticNoise, NullNoise],
        sigma_sq: SigmaSq,
    ) -> Callable:
        opt_fn = eps.apply(var_fn, "eps")
        opt_fn = sigma_sq_apply(opt_fn, sigma_sq)
        return opt_fn
