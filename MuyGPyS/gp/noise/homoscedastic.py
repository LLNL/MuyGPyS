# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""

from typing import Callable, Optional, Tuple, Union

import MuyGPyS._src.math as mm

from MuyGPyS._src.gp.noise import _homoscedastic_perturb
from MuyGPyS.gp.hyperparameter import ScalarParam
from MuyGPyS.gp.noise.noise_fn import NoiseFn


class HomoscedasticNoise(ScalarParam, NoiseFn):
    """
    A scalar prior noise parameter.

    A homoscedastic noise parameter used to build the "nugget" with the prior
    assumption that all observations are subject to i.i.d. unbiased Gaussian
    noise. Can be set at initialization time or left subject to optimization, in
    which case (positive) bounds are specified.

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
        _backend_fn: Callable = _homoscedastic_perturb,
    ):
        super(HomoscedasticNoise, self).__init__(val, bounds)
        if self.fixed() is False:
            if self._bounds[0] < 0.0 or self._bounds[1] < 0.0:
                raise ValueError(
                    f"Homoscedastic noise optimization bounds {self._bounds} "
                    f"are not strictly positive!"
                )

        self._perturb_fn = _backend_fn

    def perturb(
        self, K: mm.ndarray, noise: Optional[float] = None, **kwargs
    ) -> mm.ndarray:
        """
        Perturb a kernel tensor with homoscedastic noise.

        Applies a homoscedastic noise model to a kernel tensor, whose last two
        dimensions are assumed to be the same length. For each such square
        submatrix :math:`K`, computes the form :math:`K + \\tau^2 * I`,
        where :math:`\\tau^2` is the shared noise prior variance and
        :math:`I` is the conforming identity matrix.

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count)`-shaped kernel matrices corresponding
                to each of the batch elements.
            noise:
                A floating-point value for the noise variance prior, or `None`.
                `None` prompts the use of the stored value, whereas supplying
                alternative values is employed during optimization.

        Returns:
            A tensor of shape `(batch_count, nn_count, nn_count)` where the
            final two dimensions consist of the perturbed matrices of the input
            :math:`K`.
        """
        if noise is None:
            noise = self._val
        return self._perturb_fn(K, noise)

    def perturb_fn(self, fn: Callable) -> Callable:
        """
        Perturb a function of kernel tensors with homoscedastic noise.

        Applies a homoscedastic noise model to the first argument of the given
        function, which is assumed to be a kernel tensor whose last two
        dimensions are the same length. The returned function is the same as
        the input, save that it perturbs any passed kernel tensors.

        Args:
            fn:
                A callable whose first argument is assumed to be a tensor of
                shape `(batch_count, nn_count, nn_count)` containing the
                `(nn_count, nn_count)`-shaped kernel matrices corresponding to
                each of the batch elements.

        Returns:
            A Callable with the same signature that applies a homoscedastic
            perturbation to its first argument. Also adds a `noise` keyword
            argument that is only used for optimization.
        """

        def perturbed_fn(K, *args, noise=None, **kwargs):
            return fn(self.perturb(K, noise=noise), *args, **kwargs)

        return perturbed_fn
