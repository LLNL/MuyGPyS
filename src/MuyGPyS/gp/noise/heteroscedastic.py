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

from MuyGPyS._src.gp.noise import _heteroscedastic_perturb
from MuyGPyS.gp.hyperparameter import TensorParam
from MuyGPyS.gp.noise.noise_fn import NoiseFn


class HeteroscedasticNoise(TensorParam, NoiseFn):
    """
    A tensor noise parameter.

    A heteroscedastic noise tensor used to build the "nugget" with the prior
    assumption that all observations are have a corresponding measurement noise
    prior variance.

    Args:
        val:
            An ndarray of shape `(batch_count,) + in_shape`
            containing the heteroscedastic nugget matrix.
    Raises:
        ValueError:
            Any strictly negative entry in the array will produce an error.
    """

    def __init__(
        self, val: mm.ndarray, _backend_fn: Callable = _heteroscedastic_perturb
    ):
        super(HeteroscedasticNoise, self).__init__(val)
        if (self._val.flatten() < 0).sum() > 0:
            raise ValueError(
                "Heteroscedastic noise values are not strictly non-negative!"
            )
        self._perturb_fn = _backend_fn

    def perturb(self, Kin: mm.ndarray, **kwargs) -> mm.ndarray:
        """
        Perturb a kernel tensor with heteroscedastic noise.

        Applies a heteroscedastic noise model to a kernel tensor, whose last two
        dimensions are assumed to be the same length. For each such square
        submatrix :math:`Kin`, computes the form :math:`Kin + D`, where :math:`D`
        is the diagonal matrix containing the observation-wise noise priors.

        Args:
            Kin:
                A tensor of shape `(batch_count,) + in_shape + in_shape`
                containing the pairwise, possibly multivariate covariance
                among the neighborhood for each batch element.

        Returns:
            A tensor of shape `(batch_count,) + in_shape + in_shape` containing
            the perturbed instance of the input :math:`Kin`.
        """
        return self._perturb_fn(Kin, self._val)

    def perturb_fn(self, fn: Callable) -> Callable:
        """
        Perturb a function of kernel tensors with heteroscedastic noise.

        Applies a heteroscedastic noise model to the first argument of the given
        function, which is assumed to be a kernel tensor whose last two
        dimensions are the same length. The returned function is the same as
        the input, save that it perturbs any passed kernel tensors.

        Args:
            fn:
                A callable whose first argument is assumed to be a tensor of
                shape `(batch_count,) + in_shape + in_shape` pairwise, possibly
                multivariate covariance among the neighborhood for each batch
                element.

        Returns:
            A Callable with the same signature that applies a homoscedastic
            perturbation to its first argument.
        """

        def perturbed_fn(Kin, *args, **kwargs):
            return fn(self.perturb(Kin), *args, **kwargs)

        return perturbed_fn

    def fixed(self) -> bool:
        """
        Overloading fixed function to return True for heteroscedastic noise.

        Returns:
            `True` - we do not allow optimizing Heteroscedastic Noise.
        """
        return True
