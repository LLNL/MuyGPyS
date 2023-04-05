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
    >>> from MuyGPyS.gp.kernels import RBF
    >>> kernel_fn = RBF(metric = "l2")

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

from typing import Callable, List, Tuple, Union

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.kernels import _rbf_fn
from MuyGPyS.gp.distortion import (
    embed_with_distortion_model,
    IsotropicDistortion,
    NullDistortion,
)
from MuyGPyS.gp.kernels import (
    append_optim_params_lists,
    apply_hyperparameter,
    Hyperparameter,
    KernelFn,
)


class RBF(KernelFn):
    """
    The radial basis function (RBF) or squared-exponential kernel.

    The RBF kernel includes a single explicit length scale parameter
    :math:`\\ell>0`, and depends upon a distance function
    :math:`d(\\cdot, \\cdot)`.
    NOTE[bwp] We currently assume that the kernel is isotropic, so
    :math:`|\\ell| = 1`.

    The kernel is defined by

    .. math::
        K(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)}{2\\ell^2}\\right).

    Typically, :math:`d(\\cdot,\\cdot)` is the squared Euclidean distance
    or second frequency moment of the difference of the operands.

    Args:
        metric:
            The distance function to be used. Includes length_scale
            hyperparameter information via the MuyGPyS.gp.distortion module
    """

    def __init__(
        self,
        metric: Union[
            IsotropicDistortion, NullDistortion
        ] = IsotropicDistortion("F2", length_scale=Hyperparameter(1.0)),
    ):
        super().__init__(metric=metric)
        self._fn = _rbf_fn
        self._fn = embed_with_distortion_model(self._fn, self._distortion_fn)

    def __call__(self, diffs: mm.ndarray) -> mm.ndarray:
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
        return self._fn(diffs)

    def get_optim_params(
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
        names: List[str] = []
        params: List[float] = []
        bounds: List[Tuple[float, float]] = []
        return names, params, bounds

    def get_opt_fn(self) -> Callable:
        """
        Return a kernel function with fixed parameters set.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` and assumes
        that optimization parameters will be passed as keyword arguments.

        Returns:
            A function implementing the kernel where all fixed parameters are
            set. The function expects keyword arguments corresponding to current
            hyperparameter values for unfixed parameters.
        """
        return self._get_opt_fn(self._fn, self._distortion_fn)

    @staticmethod
    def _get_opt_fn(
        rbf_fn: KernelFn,
        distortion_fn: Union[IsotropicDistortion, NullDistortion],
    ) -> Callable:
        return super()._get_opt_fn(rbf_fn, distortion_fn)
