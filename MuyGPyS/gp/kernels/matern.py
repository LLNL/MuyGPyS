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
    ...     smoothness=Parameter("log_sample", (0.1, 2.5)),
    ...     deformation=Isotropy(
    ...         metric=l2,
    ...         length_scale=Parameter(1.0),
    ...     ),
    ... )

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

from typing import Callable, List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.kernels import (
    _matern_05_fn,
    _matern_15_fn,
    _matern_25_fn,
    _matern_inf_fn,
    _matern_gen_fn,
)

from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation import (
    DeformationFn,
    Isotropy,
    l2,
)
from MuyGPyS.gp.hyperparameter import ScalarParam
from MuyGPyS.gp.kernels import KernelFn


def _set_matern_fn(
    smoothness: ScalarParam,
    _backend_05_fn: Callable = _matern_05_fn,
    _backend_15_fn: Callable = _matern_15_fn,
    _backend_25_fn: Callable = _matern_25_fn,
    _backend_inf_fn: Callable = _matern_inf_fn,
    _backend_gen_fn: Callable = _matern_gen_fn,
):
    if smoothness.fixed() is True:
        if smoothness() == 0.5:
            return _backend_05_fn
        elif smoothness() == 1.5:
            return _backend_15_fn
        elif smoothness() == 2.5:
            return _backend_25_fn
        elif smoothness() == mm.inf:
            return _backend_inf_fn
        else:
            return _backend_gen_fn

    return _backend_gen_fn


@auto_str
class Matern(KernelFn):
    """
    The Matérn kernel.

    The Màtern kernel includes a parameterized deformation model
    :math:`d_\\ell(\\cdot, \\cdot)` and an additional smoothness parameter
    :math:`\\nu>0`. :math:`\\nu` is proportional to the smoothness of the
    resulting function. As :math:`\\nu\\rightarrow\\infty`, the kernel becomes
    equivalent to the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn
    kernel is identical to the absolute exponential kernel. Important
    intermediate values are :math:`\\nu=1.5` (once differentiable functions) and
    :math:`\\nu=2.5` (twice differentiable functions).

    The kernel is defined by

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d_\\ell(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg),

    where :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.
    Typically, :math:`d(\\cdot,\\cdot)` is the Euclidean distance or
    :math:`\\ell_2` norm of the difference of the operands.

    Args:
        smoothness:
            A parameter determining the differentiability of the function
            distribution.
        deformation:
            The deformation functor to be used. Includes length_scale
            hyperparameter information via the `MuyGPyS.gp.deformation` module.
    """

    def __init__(
        self,
        smoothness: ScalarParam = ScalarParam(0.5),
        deformation: DeformationFn = Isotropy(
            l2, length_scale=ScalarParam(1.0)
        ),
        **_backend_fns
    ):
        super().__init__(deformation=deformation)
        self.smoothness = smoothness
        self._backend_fns = _backend_fns
        self._make()

    def _make(self):
        super()._make_base()
        self._hyperparameters["smoothness"] = self.smoothness
        self._kernel_fn = _set_matern_fn(self.smoothness, **self._backend_fns)
        self._fn = self.smoothness.apply_fn(self._kernel_fn, "smoothness")
        self._fn = self.deformation.embed_fn(self._fn)

    def __call__(self, diffs, **kwargs):
        """
        Compute Matern kernels from distance tensor.

        Takes inspiration from
        `scikit-learn's implementation
        <https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/gaussian_process/kernels.py#L1529>`_.

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
        self.smoothness.append_lists("smoothness", names, params, bounds)
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
