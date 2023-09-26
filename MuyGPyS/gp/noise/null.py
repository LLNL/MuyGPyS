# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
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

    def perturb(self, K: mm.ndarray, **kwargs) -> mm.ndarray:
        """
        Null noise perturbation.

        Simply returns the input tensor unchanged.

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count)`-shaped kernel matrices corresponding
                to each of the batch elements.

        Returns:
            The same tensor.
        """
        return K

    def perturb_fn(self, fn: Callable) -> Callable:
        return fn
