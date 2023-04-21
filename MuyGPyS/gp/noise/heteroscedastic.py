# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""
import MuyGPyS._src.math as mm

from MuyGPyS.gp.hyperparameter import TensorHyperparameter


class HeteroscedasticNoise(TensorHyperparameter):
    """
    A tensor :math:`\\eps` noise parameter.

    :math:`\\epse` is a heteroscedastic noise tensor used to build the "nugget"
    with the prior assumption that all observations are have a corresponding
    measurement noise.

    Args:
        val:
            An ndarray of shape `(batch_count, nn_count)`
            containing the heteroscedastic nugget matrix.
    Raises:
        ValueError:
            Any strictly negative entry in the array will produce an error.
    """

    def __init__(
        self,
        val: mm.ndarray,
    ):
        super(HeteroscedasticNoise, self).__init__(val)
        if mm.sum(self._val.flatten() < 0) > 0:
            raise ValueError(
                "Heteroscedastic noise values are not strictly non-negative!"
            )

    def fixed(self) -> bool:
        """
        Overloading fixed function to return True for heteroscedastic noise.

        Returns:
            `True` - we do not allowed
        """
        return True
