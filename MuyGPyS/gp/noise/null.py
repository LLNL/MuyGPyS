# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""

from typing import Callable

import MuyGPyS._src.math as mm

from MuyGPyS.gp.hyperparameter import ScalarParam
from MuyGPyS.gp.noise.noise_fn import NoiseFn


class NullNoise(ScalarParam, NoiseFn):
    """
    A zero noise assumption model.
    """

    def __init__(self, *args, **kwargs):
        self.val = 0.0
        self.bounds = "fixed"

    def __call__(self, *args, **kwargs):
        return 0.0

    def perturb(self, Kin: mm.ndarray, **kwargs) -> mm.ndarray:
        """
        Null noise perturbation.

        Simply returns the input tensor unchanged.

        Args:
            Kin:
                A tensor of shape `(batch_count,) + in_shape + in_shape`
                containing the pairwise, possibly multivariate covariance
                among the neighborhood for each batch element.

        Returns:
            The same tensor.
        """
        return Kin

    def perturb_fn(self, fn: Callable) -> Callable:
        return fn
