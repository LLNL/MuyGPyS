# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Hyperparameters and kernel functors

Defines the Matérn kernel functor (inheriting
:class:`~MuyGPyS.gp.kernels.kernel_fn.KernelFn`) that transform crosswise and
pairwise difference tensors into cross-covariance and covariance (or kernel)
tensors, respectively.

See the following example to initialize an :class:`MuyGPyS.gp.kernels.Matern`
object.

Example:
    >>> from MuyGPyS.gp.kernels import Matern
    >>> kern = Matern(
    ...         nu = {"val": "log_sample", "bounds": (0.1, 2.5)},
    ...         length_scale = {"val": 7.2},
    ...         metric = "l2",
    ... }

One uses a previously computed `pairwise_diffs` tensor (see
:func:`MuyGPyS.gp.tensors.pairwise_tensor`) to compute a kernel tensor whose
second two dimensions contain square kernel matrices. Similarly, one uses a
previously computed `crosswise_diffs` matrix (see
:func:`MuyGPyS.gp.tensor.crosswise_tensor`) to compute a cross-covariance
matrix. See the following example, which assumes that you have already
constructed the differenece tensors and kernel as shown above.

Example:
    >>> K = kern(pairwise_diffs)
    >>> Kcross = kern(crosswise_diffs)
"""

from typing import Callable, List, Tuple, Union

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.kernels import (
    _matern_05_fn,
    _matern_15_fn,
    _matern_25_fn,
    _matern_inf_fn,
    _matern_gen_fn,
)
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


def _set_matern_fn(nu: Hyperparameter):
    if nu.fixed() is True:
        if nu() == 0.5:
            return _matern_05_fn
        elif nu() == 1.5:
            return _matern_15_fn
        elif nu() == 2.5:
            return _matern_25_fn
        elif nu() == mm.inf:
            return _matern_inf_fn
        else:

            return _matern_gen_fn

    return _matern_gen_fn


class Matern(KernelFn):
    """
    The Màtern kernel.

    The Màtern kernel includes a length scale parameter :math:`\\ell>0` and an
    additional smoothness parameter :math:`\\nu>0`. :math:`\\nu` is inversely
    proportional to the smoothness of the resulting function. The Màtern kernel
    also depends upon a distance function :math:`d(\\cdot, \\cdot)`.
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to
    the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn kernel
    becomes identical to the absolute exponential kernel.
    Important intermediate values are
    :math:`\\nu=1.5` (once differentiable functions)
    and :math:`\\nu=2.5` (twice differentiable functions).
    NOTE[bwp] We currently assume that the kernel is isotropic, so
    :math:`|\\ell| = 1`.

    The kernel is defined by

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg),

    where :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.

    Typically, :math:`d(\\cdot,\\cdot)` is the Euclidean distance or
    :math:`\\ell_2` norm of the difference of the operands.

    Args:
        nu:
            A hyperparameter dict defining the length_scale parameter.
        length_scale:
            A hyperparameter dict defining the length_scale parameter.
        metric:
            The distance function to be used.
    """

    def __init__(
        self,
        nu: Hyperparameter = Hyperparameter(0.5),
        length_scale: Hyperparameter = Hyperparameter(1.0),
        metric: Union[
            IsotropicDistortion, NullDistortion
        ] = IsotropicDistortion("l2"),
    ):
        super().__init__(metric=metric)
        self.nu = nu
        self.length_scale = length_scale
        self.hyperparameters["nu"] = self.nu
        self.hyperparameters["length_scale"] = self.length_scale
        self._fn = _set_matern_fn(self.nu)
        self._fn = embed_with_distortion_model(self._fn, self._distortion_fn)

    def __call__(self, diffs):
        """
        Compute Matern kernels from distance tensor.

        Takes inspiration from
        [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/gaussian_process/kernels.py#L1529)

        Args:
            diffs:
                A tensor of pairwise differences of shape
                `(data_count, nn_count, nn_count, feature_count)`. It is assumed
                that the vectors along the diagonals diffs[i, j, j, :] == 0.

        Returns:
            A cross-covariance matrix of shape `(data_count, nn_count)` or a
            tensor of shape `(data_count, nn_count, nn_count)` whose last two
            dimensions are kernel matrices.
        """
        return self._fn(diffs, nu=self.nu(), length_scale=self.length_scale())

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
        append_optim_params_lists(self.nu, "nu", names, params, bounds)
        append_optim_params_lists(
            self.length_scale, "length_scale", names, params, bounds
        )
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
        return self._get_opt_fn(self._fn, self.nu, self.length_scale)

    @staticmethod
    def _get_opt_fn(
        matern_fn: Callable, nu: Hyperparameter, length_scale: Hyperparameter
    ) -> Callable:
        opt_fn = apply_hyperparameter(matern_fn, length_scale, "length_scale")
        opt_fn = apply_hyperparameter(opt_fn, nu, "nu")
        return opt_fn
