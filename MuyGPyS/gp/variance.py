# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import _muygps_diagonal_variance
from MuyGPyS.gp.hyperparameter import ScaleFn
from MuyGPyS.gp.noise import NoiseFn


# NOTE: Should we pull scale out of here and put it into kernel function? It
# really doesn't belong here.


class PosteriorVariance:
    def __init__(
        self,
        Kout: mm.ndarray,
        noise: NoiseFn,
        scale: ScaleFn,
        _backend_fn: Callable = _muygps_diagonal_variance,
    ):
        self._fn = _backend_fn
        self._fn = noise.perturb_fn(self._fn)

        def fix_Kout_fn(fn: Callable) -> Callable:
            def fixed_Kout_fn(Kin, Kcross, *args, **kwargs):
                return fn(Kin, Kcross, Kout, *args, **kwargs)

            return fixed_Kout_fn

        self._fn = fix_Kout_fn(self._fn)
        self._opt_fn = self._fn
        self._fn = scale.scale_fn(self._fn)

    def __call__(
        self,
        Kin: mm.ndarray,
        Kcross: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        return self._fn(Kin, Kcross, **kwargs)

    def get_opt_fn(self) -> Callable:
        return self._opt_fn
