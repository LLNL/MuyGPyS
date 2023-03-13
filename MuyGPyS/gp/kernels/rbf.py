# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
RBF kernel functor

Defines RBF (or Gaussian, or squared exponential) kernel  (inheriting
:class:`~MuyGPyS.gp.kernels.kernel_fn.KernelFn`) that transform crosswise
distance matrices into cross-covariance matrices and pairwise distance matrices
into covariance or kernel matrices.

See the following example to initialize an :class:`MuyGPyS.gp.kernels.Matern`
object. Other kernel functors are similar, but require different
hyperparameters.

Example:
    >>> from MuyGPyS.gp.kernels import RBF
    >>> kern = RBF(
    ...         length_scale = {"val": 7.2, , "bounds": (0.1, 2.5)}},
    ...         metric = "l2",
    ... }

One uses a previously computed `pairwise_dists` tensor (see
:func:`MuyGPyS.gp.distance.pairwise_distance`) to compute a kernel tensor whose
second two dimensions contain square kernel matrices. Similarly, one uses a
previously computed `crosswise_dists` matrix (see
:func:`MuyGPyS.gp.distance.crosswise_distance`) to compute a cross-covariance
matrix. See the following example, which assumes that you have already
constructed the distance `numpy.nparrays` and the kernel `kern` as shown above.

Example:
    >>> K = kern(pairwise_dists)
    >>> Kcross = kern(crosswise_dists)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.kernels import _rbf_fn
from MuyGPyS.gp.kernels.hyperparameters import (
    _init_hyperparameter,
    Hyperparameter,
)
from MuyGPyS.gp.kernels.kernel_fn import KernelFn


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
        length_scale:
            A hyperparameter dict defining the length_scale parameter.
        metric:
            The distance function to be used. Defaults to `"F2"`.
    """

    def __init__(
        self,
        length_scale: Dict[
            str, Union[str, float, Tuple[float, float]]
        ] = dict(),
        metric: Optional[str] = "F2",
    ):
        super().__init__()
        self.length_scale = _init_hyperparameter(1.0, "fixed", **length_scale)
        self.hyperparameters["length_scale"] = self.length_scale
        self.metric = metric

    def __call__(self, squared_dists: mm.ndarray) -> mm.ndarray:
        """
        Compute RBF kernel(s) from a distance matrix or tensor.

        Args:
            squared_dists:
                A matrix or tensor of pairwise distances (usually squared l2 or
                F2) of shape `(data_count, nn_count, nn_count)` or
                `(data_count, nn_count)`. In the tensor case, matrix diagonals
                along last two dimensions are expected to be 0.

        Returns:
            A cross-covariance matrix of shape `(data_count, nn_count)` or a
            tensor of shape `(data_count, nn_count, nn_count)` whose last two
            dimensions are kernel matrices.
        """
        return self._fn(squared_dists, length_scale=self.length_scale())

    @staticmethod
    def _fn(squared_dists: mm.ndarray, length_scale: float) -> mm.ndarray:
        return _rbf_fn(squared_dists, length_scale)

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
        names = []
        params = []
        bounds = []
        if not self.length_scale.fixed():
            names.append("length_scale")
            params.append(self.length_scale())
            bounds.append(self.length_scale.get_bounds())
        return names, params, bounds

    def get_array_opt_fn(self) -> Callable:
        """
        Return a kernel function with fixed parameters set.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` with
        `opt_method="scipy"`, and assumes that the optimization parameters will
        be passed in an `(optim_count,)` vector.

        Returns:
            A function implementing the kernel where all fixed parameters are
            set. The function expects a list of current hyperparameter values
            for unfixed parameters, which are expected to occur in a certain
            order matching how they are set in
            :func:`~MuyGPyS.gp.kernel.RBF.get_optim_params()`.
        """
        return self._get_array_opt_fn(_rbf_fn, self.length_scale)

    @staticmethod
    def _get_array_opt_fn(
        rbf_fn: Callable, length_scale: Hyperparameter
    ) -> Callable:
        if not length_scale.fixed():

            def caller_fn(dists, x0):
                return rbf_fn(dists, length_scale=x0[0])

        else:

            def caller_fn(dists, x0):
                return rbf_fn(dists, length_scale=length_scale())

        return caller_fn

    def get_kwargs_opt_fn(self) -> Callable:
        """
        Return a kernel function with fixed parameters set.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` with
        `opt_method="bayesian"`, and assumes that optimization parameters will
        be passed as keyword arguments.

        Returns:
            A function implementing the kernel where all fixed parameters are
            set. The function expects keyword arguments corresponding to current
            hyperparameter values for unfixed parameters.
        """
        return self._get_kwargs_opt_fn(_rbf_fn, self.length_scale)

    @staticmethod
    def _get_kwargs_opt_fn(
        rbf_fn: Callable, length_scale: Hyperparameter
    ) -> Callable:
        if not length_scale.fixed():

            def caller_fn(dists, **kwargs):
                return rbf_fn(dists, length_scale=kwargs["length_scale"])

        else:

            def caller_fn(dists, **kwargs):
                return rbf_fn(dists, length_scale=length_scale())

        return caller_fn
