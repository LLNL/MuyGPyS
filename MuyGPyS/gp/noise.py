# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""

import numpy as np

from typing import cast, Callable, Dict, List, Optional, Tuple, Union

from MuyGPyS import config

from MuyGPyS._src.gp.kernels import (
    _rbf_fn,
    _matern_05_fn,
    _matern_15_fn,
    _matern_25_fn,
    _matern_inf_fn,
    _matern_gen_fn,
)
from MuyGPyS._src.mpi_utils import _is_mpi_mode
from MuyGPyS.optimize.utils import _switch_on_opt_method

from MuyGPyS.gp.kernels import Hyperparameter


class HomoscedasticNoise(Hyperparameter):
    """
    A scalar :math:`\\eps` prior noise parameter.

    :math:`\\epse` is a homoscedastic noise parameter used to build the "nugget"
    with the prior assumption that all observations are subject to i.i.d.
    Gaussian noise. Can be set at initialization time or left subject to
    optimization, in which case (positive) bounds are specified.

    Args:
        val:
            A positive scalar, or the strings `"sample"` or `"log_sample"`.
        bounds:
            Iterable container of len 2 containing positive lower and upper
            bounds (in that order), or the string `"fixed"`.

    Raises:
        ValueError:
            Any nonpositive `bounds` string will produce an error.
    """

    def __init__(
        self,
        val: Union[str, float],
        bounds: Union[str, Tuple[float, float]],
    ):
        super(HomoscedasticNoise, self).__init__(val, bounds)
        if self.fixed() is False:
            if self._bounds[0] < 0.0 or self._bounds[1] < 0.0:
                raise ValueError(
                    f"Homoscedastic noise optimization bounds {self._bounds} "
                    f"are not strictly positive!"
                )
