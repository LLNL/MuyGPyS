# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""

from typing import Tuple, Union

from MuyGPyS.gp.hyperparameter import ScalarHyperparameter


class HomoscedasticNoise(ScalarHyperparameter):
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
        bounds: Union[str, Tuple[float, float]] = "fixed",
    ):
        super(HomoscedasticNoise, self).__init__(val, bounds)
        if self.fixed() is False:
            if self._bounds[0] < 0.0 or self._bounds[1] < 0.0:
                raise ValueError(
                    f"Homoscedastic noise optimization bounds {self._bounds} "
                    f"are not strictly positive!"
                )
