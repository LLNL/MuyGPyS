# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Shear kernel functor

Defines the lensing shear kernel (inheriting
:class:`~MuyGPyS.gp.kernels.kernel_fn.KernelFn`) that transform crosswise and
pairwise difference tensors into cross-covariance and covariance (or kernel)
tensors, respectively.

This kernel is defined in terms of second-order partial derivatives of the RBF
kernel. Mathematical derivation is forthcoming.

Example:
    >>> shear_fn = ShearKenrel(
    ...     deformation=Isotropy(
    ...         F2,
    ...         length_scale=ScalarParam(1.0),
    ...     ),
    ... )

One uses a previously computed `pairwise_diffs` tensor (see
:func:`MuyGPyS.gp.tensors.pairwise_tensor`) to compute a kernel tensor whose
second two dimensions contain square kernel matrices. Similarly, one uses a
previously computed `crosswise_diffs` matrix (see
:func:`MuyGPyS.gp.tensors.crosswise_tensor`) to compute a cross-covariance
matrix. See the following example, which assumes that you have already
constructed the difference `numpy.nparrays` and the kernel `kernel_fn` as shown
above.

Example:
    >>> K = shear_fn(pairwise_diffs)
    >>> Kcross = shear_fn(crosswise_diffs)

Note that the :class:`~MuyGPyS.gp.kernels.experimental.shear.ShearKernel`
functor
"""

from typing import Callable, List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.kernels.shear import _shear_fn
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation import (
    Isotropy,
    F2,
)
from MuyGPyS.gp.kernels import KernelFn
from MuyGPyS.gp.hyperparameter import ScalarParam


@auto_str
class ShearKernel(KernelFn):
    """
    The lensing shear kernel.

    The lensing shear kernel is defined in terms of second-order partial
    derivates of the RBF kernel.

    The kernel is defined by

    .. math::
        K(x_i, x_j) = \\dots

    Args:
        deformation:
            The deformation function to be used. Includes length_scale
            hyperparameter information via the MuyGPyS.gp.deformation module
    """

    def __init__(
        self,
        deformation: Isotropy = Isotropy(F2, length_scale=ScalarParam(1.0)),
        _backend_fn: Callable = _shear_fn,
    ):
        super().__init__(deformation=deformation)
        if not isinstance(self.deformation, Isotropy):
            raise ValueError(
                "ShearKernel only supports isotropic deformations, not "
                f"{type(deformation)}"
            )
        self._kernel_fn = _backend_fn
        self._make()

    def _make(self):
        super()._make_base()
        self._fn = self._kernel_fn

    def __call__(self, diffs: mm.ndarray, **kwargs) -> mm.ndarray:
        """
        Compute shear kernel(s) from a difference tensor.

        Args:
            diffs:
                A tensor of pairwise diffs of shape
                `(data_count, nn_count, nn_count, feature_count)` or
                `(data_count, nn_count, feature_count)`. In the four dimensional
                case, it is assumed that the diagonals dists
                diffs[i, j, j, :] == 0.

        Returns:
            A cross-covariance matrix of shape `(data_count * 3, nn_count * 3)`
            or a tensor of shape `(data_count, nn_count * 3, nn_count * 3)`
            whose last two dimensions are kernel matrices.
        """
        return self._fn(diffs, **kwargs)

    def get_opt_params(
        self,
    ) -> Tuple[List[str], List[float], List[Tuple[float, float]]]:
        """
        Report lists of unfixed hyperparameter names, values, and bounds.

        Returns
        -------
            names:
                A list of unfixed hyperparameter names.
            params:
                A list of unfixed hyperparameter values.
            bounds:
                A list of unfixed hyperparameter bound tuples.
        """
        names, params, bounds = super().get_opt_params()
        return names, params, bounds

    def get_opt_fn(self) -> Callable:
        """
        Return a kernel function with fixed parameters set.

        Assumes that optimization parameter literals will be passed as keyword
        arguments.

        Returns:
            A function implementing the kernel where all fixed parameters are
            set. The function expects keyword arguments corresponding to current
            hyperparameter values for unfixed parameters.
        """
        return self._fn
