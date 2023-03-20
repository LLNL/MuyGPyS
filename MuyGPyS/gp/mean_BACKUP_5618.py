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
<<<<<<< HEAD
from MuyGPyS._src.gp.noise import (
    _homoscedastic_perturb,
    _heteroscedastic_perturb,
)
from MuyGPyS.gp.kernels import apply_hyperparameter
from MuyGPyS.gp.noise import (
    HomoscedasticNoise,
    HeteroscedasticNoise,
    noise_perturb,
)
=======
from MuyGPyS.gp.kernels import apply_hyperparameter
from MuyGPyS.gp.noise import HomoscedasticNoise, perturb_with_noise_model
>>>>>>> develop


class PosteriorMean:
    def __init__(
        self, eps: Union[HomoscedasticNoise, HeteroscedasticNoise], **kwargs
    ):
        self.eps = eps
        self._fn = _muygps_posterior_mean
<<<<<<< HEAD
        if isinstance(self.eps, HomoscedasticNoise):
            self._fn = noise_perturb(_homoscedastic_perturb)(self._fn)
        elif isinstance(self.eps, HeteroscedasticNoise):
            self._fn = noise_perturb(_heteroscedastic_perturb)(self._fn)
        else:
            raise ValueError(f"Noise model {type(self.eps)} is not supported")
=======
        self._fn = perturb_with_noise_model(self._fn, self.eps)
>>>>>>> develop

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
        mean_fn: Callable, eps: Union[HomoscedasticNoise, HeteroscedasticNoise]
    ) -> Callable:
        opt_fn = apply_hyperparameter(mean_fn, eps, "eps")
        return opt_fn
