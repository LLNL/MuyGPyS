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
from MuyGPyS.gp.hyperparameter import ScaleFn
from MuyGPyS.gp.noise import NoiseFn


class PosteriorVariance:
    def __init__(
        self,
        noise: NoiseFn,
        scale: ScaleFn,
        apply_scale: bool = True,
        _backend_fn: Callable = _muygps_diagonal_variance,
    ):
        self._fn = _backend_fn
        self._fn = noise.perturb_fn(self._fn)
        if apply_scale is True:
            self._fn = scale.scale_fn(self._fn)

    def __call__(
        self,
        K: mm.ndarray,
        Kcross: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        if len(Kcross.shape) == 2:
            batch_count, nn_count = Kcross.shape
            Kcross = Kcross.reshape(batch_count, 1, nn_count)
        variances = self._fn(K, Kcross, **kwargs)
        if len(variances.shape) == 1:
            variances = variances.reshape(variances.shape[0], 1)
        return variances

    def get_opt_fn(self) -> Callable:
        return self.__call__
