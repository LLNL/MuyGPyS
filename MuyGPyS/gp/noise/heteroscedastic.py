# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""
import MuyGPyS._src.math as mm

from MuyGPyS.gp.kernels import Hyperparameter


class HeteroscedasticNoise(Hyperparameter):
    """
    A tensor :math:`\\eps` noise parameter.

    :math:`\\epse` is a heteroscedastic noise tensor used to build the "nugget"
    with the prior assumption that all observations are have a corresponding
    measurement noise.

    Args:
        val:
            An ndarray of shape `(batch_count, nn_count, nn_count)`
            containing the heteroscedastic nugget matrix.
        bounds:
            Must be set to the string "fixed" for now. We do not support
            the training of individual measurement noise values in the
            current model.

    Raises:
        ValueError:
            Any strictly negative entry in the array will produce an error.
    """

    def __init__(
        self,
        val: mm.ndarray,
        bounds: str,
    ):
        super(HeteroscedasticNoise, self).__init__(val, bounds)
        self.bounds = "fixed"
        self.val = val
        if mm.sum(self.val < 0) > 0:
            raise ValueError(
                f"Heteroscedastic noise values are not strictly non-negative!"
            )
