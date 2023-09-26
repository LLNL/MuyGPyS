# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
RBF kernel functor

Defines RBF (or Gaussian, or squared exponential) kernel  (inheriting
:class:`~MuyGPyS.gp.kernels.kernel_fn.KernelFn`) that transform crosswise and
pairwise difference tensors into cross-covariance and covariance (or kernel)
tensors, respectively.

See the following example to initialize an :class:`MuyGPyS.gp.kernels.Matern`
object. Other kernel functors are similar, but require different
hyperparameters.

Example:
    >>> kernel_fn = RBF(
    ...     deformation=Isotropy(
    ...         metric=l2,
    ...         length_scale=Parameter(1.0),
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
    >>> K = kernel_fn(pairwise_diffs)
    >>> Kcross = kernel_fn(crosswise_diffs)
"""

from typing import Callable

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.kernels import _rbf_fn
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation import DeformationFn, Isotropy, F2
from MuyGPyS.gp.kernels import KernelFn
from MuyGPyS.gp.hyperparameter import ScalarParam


@auto_str
class RBF(KernelFn):
    """
    The radial basis function (RBF) or squared-exponential kernel.

    The RBF kernel includes a parameterized scaled distance function
    :math:`d_\\ell(\\cdot, \\cdot)`.

    The kernel is defined by

    .. math::
        K(x_i, x_j) = \\exp\\left(- d_\\ell(x_i, x_j)\\right).

    Typically, :math:`d(\\cdot,\\cdot)` is the squared Euclidean distance
    or second frequency moment of the difference of the operands.

    Args:
        deformation:
            The deformation functor to be used. Includes length_scale
            hyperparameter information via the `MuyGPyS.gp.deformation` module
    """

    def __init__(
        self,
        deformation: DeformationFn = Isotropy(
            F2, length_scale=ScalarParam(1.0)
        ),
        _backend_fn: Callable = _rbf_fn,
    ):
        super().__init__(deformation=deformation)
        self._kernel_fn = _backend_fn
        self._make()

    def _make(self):
        super()._make_base()
        self._fn = self.deformation.embed_fn(self._kernel_fn)

    def __call__(self, diffs: mm.ndarray, **kwargs) -> mm.ndarray:
        """
        Compute RBF kernel(s) from a difference tensor.

        Args:
            diffs:
                A tensor of pairwise diffs of shape
                `(data_count, nn_count, nn_count, feature_count)` or
                `(data_count, nn_count, feature_count)`. In the four dimensional
                case, it is assumed that the diagonals dists
                diffs[i, j, j, :] == 0.

        Returns:
            A cross-covariance matrix of shape `(data_count, nn_count)` or a
            tensor of shape `(data_count, nn_count, nn_count)` whose last two
            dimensions are kernel matrices.
        """
        return self._fn(diffs, **kwargs)

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
