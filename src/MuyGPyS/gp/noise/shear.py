# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""

from typing import Callable, Tuple, Union

from MuyGPyS._src.gp.noise import _shear_perturb33

from MuyGPyS.gp.noise.homoscedastic import HomoscedasticNoise


class ShearNoise33(HomoscedasticNoise):

    def __init__(
        self,
        val: Union[str, float],
        bounds: Union[str, Tuple[float, float]] = "fixed",
        _backend_fn: Callable = _shear_perturb33,
    ):
        super(ShearNoise33, self).__init__(val, bounds, _backend_fn=_backend_fn)
