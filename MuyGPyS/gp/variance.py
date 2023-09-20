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
from MuyGPyS.gp.sigma_sq import SigmaSq
from MuyGPyS.gp.noise import NoiseFn


class PosteriorVariance:
    def __init__(
        self,
        eps: NoiseFn,
        sigma_sq: SigmaSq,
        apply_sigma_sq: bool = True,
        _backend_fn: Callable = _muygps_diagonal_variance,
    ):
        self._fn = _backend_fn
        self._fn = eps.perturb_fn(self._fn)
        if apply_sigma_sq is True:
            self._fn = sigma_sq.scale_fn(self._fn)

    def __call__(
        self,
        K: mm.ndarray,
        Kcross: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        return self._fn(K, Kcross, **kwargs)

    def get_opt_fn(self) -> Callable:
        return self._fn
