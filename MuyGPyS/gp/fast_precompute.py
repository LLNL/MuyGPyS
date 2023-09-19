# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable, Union

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import (
    _muygps_fast_posterior_mean_precompute,
)
from MuyGPyS.gp.noise import HomoscedasticNoise, HeteroscedasticNoise, NullNoise


class FastPrecomputeCoefficients:
    def __init__(
        self,
        eps: Union[HeteroscedasticNoise, HomoscedasticNoise, NullNoise],
        _backend_fn: Callable = _muygps_fast_posterior_mean_precompute,
        **kwargs,
    ):
        self._fn = _backend_fn
        self._fn = eps.perturb_fn(self._fn)

    def __call__(
        self,
        K: mm.ndarray,
        train_nn_targets_fast: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        return self._fn(K, train_nn_targets_fast, **kwargs)
