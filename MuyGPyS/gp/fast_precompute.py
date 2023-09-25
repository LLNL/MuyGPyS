# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import (
    _muygps_fast_posterior_mean_precompute,
)
from MuyGPyS.gp.noise import NoiseFn


class FastPrecomputeCoefficients:
    def __init__(
        self,
        noise: NoiseFn,
        _backend_fn: Callable = _muygps_fast_posterior_mean_precompute,
        **kwargs,
    ):
        self._fn = _backend_fn
        self._fn = noise.perturb_fn(self._fn)

    def __call__(
        self,
        K: mm.ndarray,
        train_nn_targets_fast: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        return self._fn(K, train_nn_targets_fast, **kwargs)
