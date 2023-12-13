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
from MuyGPyS.gp.noise import NoiseFn


class PosteriorMean:
    def __init__(
        self,
        noise: NoiseFn,
        _backend_fn: Callable = _muygps_posterior_mean,
        **kwargs,
    ):
        self._fn = _backend_fn
        self._fn = noise.perturb_fn(self._fn)

    def __call__(
        self,
        Kcov: mm.ndarray,
        Kcross: mm.ndarray,
        batch_nn_targets: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        if len(Kcross.shape) == 2:
            batch_count, nn_count = Kcross.shape
            Kcross = Kcross.reshape(batch_count, 1, nn_count)
        responses = self._fn(Kcov, Kcross, batch_nn_targets, **kwargs)
        if len(responses.shape) == 1:
            responses = responses.reshape(responses.shape[0], 1)
        return responses

    def get_opt_fn(self) -> Callable:
        return self.__call__
