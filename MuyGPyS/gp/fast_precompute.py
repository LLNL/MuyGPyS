# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Union

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import (
    _muygps_fast_posterior_mean_precompute,
)
from MuyGPyS.gp.noise import (
    HomoscedasticNoise,
    HeteroscedasticNoise,
    NullNoise,
    perturb_with_noise_model,
)


class FastPrecomputeCoefficients:
    def __init__(
        self,
        eps: Union[HeteroscedasticNoise, HomoscedasticNoise, NullNoise],
        **kwargs
    ):
        self.eps = eps
        self._fn = _muygps_fast_posterior_mean_precompute
        self._fn = perturb_with_noise_model(self._fn, self.eps)

    def __call__(
        self,
        K: mm.ndarray,
        train_nn_targets_fast: mm.ndarray,
    ) -> mm.ndarray:
        return self._fn(K, train_nn_targets_fast, eps=self.eps())
