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


def _reshaper_to_be_removed(fn: Callable) -> Callable:
    def reshaped_fn(Kin, Kcross, *args, **kwargs):
        if len(Kcross.shape) == 2:
            batch_count, nn_count = Kcross.shape
            Kcross = Kcross.reshape(batch_count, 1, nn_count)
        ret = fn(Kin, Kcross, *args, **kwargs)
        if len(ret.shape) == 1:
            ret = ret.reshape(ret.shape[0], 1)
        return ret

    return reshaped_fn


class PosteriorMean:
    def __init__(
        self,
        noise: NoiseFn,
        _backend_fn: Callable = _muygps_posterior_mean,
        **kwargs,
    ):
        self._fn = _backend_fn
        self._fn = noise.perturb_fn(self._fn)
        self._fn = _reshaper_to_be_removed(self._fn)

    def __call__(
        self,
        Kin: mm.ndarray,
        Kcross: mm.ndarray,
        batch_nn_targets: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        return self._fn(Kin, Kcross, batch_nn_targets, **kwargs)

    def get_opt_fn(self) -> Callable:
        return self.__call__
