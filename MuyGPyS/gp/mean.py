# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import _muygps_posterior_mean
from MuyGPyS._src.gp.noise import _homoscedastic_perturb
from MuyGPyS.gp.kernels import apply_hyperparameter
from MuyGPyS.gp.noise import HomoscedasticNoise, noise_perturb


class PosteriorMean:
    def __init__(self, eps: HomoscedasticNoise, **kwargs):
        self.eps = eps
        self._fn = _muygps_posterior_mean
        if isinstance(self.eps, HomoscedasticNoise):
            self._fn = noise_perturb(_homoscedastic_perturb)(self._fn)
        else:
            raise ValueError(f"Noise model {type(self.eps)} is not supported")

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
    def _get_opt_fn(mean_fn: Callable, eps: HomoscedasticNoise) -> Callable:
        if isinstance(eps, HomoscedasticNoise):
            opt_fn = apply_hyperparameter(mean_fn, eps, "eps")
        else:
            raise TypeError(
                f"Noise parameter type {type(eps)} is not supported for "
                f"optimization!"
            )
        return opt_fn
