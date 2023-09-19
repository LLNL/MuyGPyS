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
from MuyGPyS.gp.sigma_sq import SigmaSq
from MuyGPyS.gp.noise import HomoscedasticNoise, HeteroscedasticNoise, NullNoise


class PosteriorVariance:
    def __init__(
        self,
        eps: Union[HomoscedasticNoise, HeteroscedasticNoise, NullNoise],
        sigma_sq: SigmaSq,
        apply_sigma_sq: bool = True,
        _backend_fn: Callable = _muygps_diagonal_variance,
    ):
        self.eps = eps
        self.sigma_sq = sigma_sq
        self._fn = _backend_fn
        self._fn = self.eps.perturb_fn(self._fn)
        if apply_sigma_sq is True:
            self._fn = sigma_sq.scale_fn(self._fn)

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
        return opt_fn
