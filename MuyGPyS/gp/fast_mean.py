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
    _muygps_fast_posterior_mean,
)


class FastPosteriorMean:
    def __init__(self, _backend_fn: Callable = _muygps_fast_posterior_mean):
        self._fn = _backend_fn

    def __call__(
        self,
        Kcross: mm.ndarray,
        coeffs_tensors: mm.ndarray,
    ) -> mm.ndarray:
        return self._fn(Kcross, coeffs_tensors)
