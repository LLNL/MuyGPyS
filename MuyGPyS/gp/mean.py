# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable, Union

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import _muygps_posterior_mean
from MuyGPyS.gp.noise import (
    HeteroscedasticNoise,
    HomoscedasticNoise,
    NullNoise,
    perturb_with_noise_model,
)


class PosteriorMean:
    def __init__(
        self,
        eps: Union[HeteroscedasticNoise, HomoscedasticNoise, NullNoise],
        **kwargs
    ):
        self.eps = eps
        self._fn = _muygps_posterior_mean
        self._fn = perturb_with_noise_model(self._fn, self.eps)

    def __call__(
        self,
        K: mm.ndarray,
        Kcross: mm.ndarray,
        batch_nn_targets: mm.ndarray,
    ) -> mm.ndarray:
        return self._fn(K, Kcross, batch_nn_targets, eps=self.eps())

    def get_opt_fn(self) -> Callable:
        return self._get_opt_fn(self._fn, self.eps)

    @staticmethod
    def _get_opt_fn(
        mean_fn: Callable,
        eps: Union[HeteroscedasticNoise, HomoscedasticNoise, NullNoise],
    ) -> Callable:
        opt_fn = eps.apply(mean_fn, "eps")
        return opt_fn
