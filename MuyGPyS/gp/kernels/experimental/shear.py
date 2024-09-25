# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
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
    >>> Kin = shear_fn(pairwise_diffs)
    >>> Kcross = shear_fn(crosswise_diffs)

Note that the :class:`~MuyGPyS.gp.kernels.experimental.shear.ShearKernel`
functor
"""

from typing import Callable, List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.kernels.shear import (
    _shear_33_fn,
    _shear_Kcross23_fn,
    _shear_Kin23_fn,
)
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation import (
    DifferenceIsotropy,
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
        Kin(x_i, x_j) = \\dots

    Args:
        deformation:
            The deformation function to be used. Includes length_scale
            hyperparameter information via the MuyGPyS.gp.deformation module
    """

    def __init__(
        self,
        deformation: DifferenceIsotropy = DifferenceIsotropy(
            F2, length_scale=ScalarParam(1.0)
        ),
        _backend_fn: Callable = _shear_33_fn,
        _backend_zeros: Callable = mm.zeros,
        _backend_squeeze: Callable = mm.squeeze,
    ):
        super().__init__(deformation=deformation)
        if not isinstance(self.deformation, DifferenceIsotropy):
            raise ValueError(
                "ShearKernel only supports the specialized difference "
                f"isotropicdeformations, not {type(deformation)}"
            )
        self._backend_zeros = _backend_zeros
        self._backend_squeeze = _backend_squeeze
        self._kernel_fn = _backend_fn
        self._make()

    def _make(self):
        super()._make_base()

        # Need length_scale passthrough
        def embedded_fn(diffs, *args, length_scale=None, **kwargs):
            if length_scale is None:
                length_scale = self.deformation.length_scale()
            return self._kernel_fn(
                diffs, *args, length_scale=length_scale, **kwargs
            )

        self._fn = embedded_fn

    def __call__(self, diffs: mm.ndarray, adjust=True, **kwargs) -> mm.ndarray:
        """
        Compute shear kernel(s) from a difference tensor.

        Args:
            diffs:
                A tensor of pairwise or crosswise distances or distances of
                shape `(data_count, nn_count, nn_count, feature_count)` or
                `(data_count, nn_count, feature_count)`.

        Returns:
            A cross-covariance matrix of shape `(data_count * 3, nn_count * 3)`
            or a tensor of shape `(data_count, nn_count * 3, nn_count * 3)`
            whose last two dimensions are kernel matrices.
        """
        if adjust and diffs.shape[-2] != diffs.shape[-3]:
            # this is probably a crosswise differences tensor
            # reshape to insert a unitary dimension
            diffs = diffs[..., None, :]
        return self._fn(diffs, **kwargs)

    def Kout(self, **kwargs) -> mm.ndarray:
        return self.__call__(self._backend_zeros((1, 1, 2)))

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
        return self.__call__


@auto_str
class ShearKernel2in3out(KernelFn):
    """
    The lensing shear kernel from observations of the two shear parameters,
    predicting both the shear parameters and the convergence.

    The lensing shear kernel is defined in terms of second-order partial
    derivates of the RBF kernel.

    The kernel is defined by

    .. math::
        Kin(x_i, x_j) = \\dots

    Args:
        deformation:
            The deformation function to be used. Includes length_scale
            hyperparameter information via the MuyGPyS.gp.deformation module
    """

    def __init__(
        self,
        deformation: DifferenceIsotropy = DifferenceIsotropy(
            F2, length_scale=ScalarParam(1.0)
        ),
        _backend_Kin_fn: Callable = _shear_Kin23_fn,
        _backend_Kcross_fn: Callable = _shear_Kcross23_fn,
        _backend_Kout_fn: Callable = _shear_33_fn,
        _backend_zeros: Callable = mm.zeros,
        _backend_squeeze: Callable = mm.squeeze,
    ):
        super().__init__(deformation=deformation)
        if not isinstance(self.deformation, DifferenceIsotropy):
            raise ValueError(
                "ShearKernel only supports the specialized difference "
                f"isotropicdeformations, not {type(deformation)}"
            )
        self._backend_zeros = _backend_zeros
        self._backend_squeeze = _backend_squeeze
        self._kernel_in_fn = _backend_Kin_fn
        self._kernel_cross_fn = _backend_Kcross_fn
        self._kernel_out_fn = _backend_Kout_fn
        self._make()

    def _make(self):
        super()._make_base()

        # Need length_scale passthrough
        def embedded_Kin_fn(diffs, *args, length_scale=None, **kwargs):
            if length_scale is None:
                length_scale = self.deformation.length_scale()
            return self._kernel_in_fn(
                diffs, *args, length_scale=length_scale, **kwargs
            )

        def embedded_Kcross_fn(diffs, *args, length_scale=None, **kwargs):
            if length_scale is None:
                length_scale = self.deformation.length_scale()
            return self._kernel_cross_fn(
                diffs, *args, length_scale=length_scale, **kwargs
            )

        def embedded_Kout_fn(diffs, *args, length_scale=None, **kwargs):
            if length_scale is None:
                length_scale = self.deformation.length_scale()
            return self._kernel_out_fn(
                diffs, *args, length_scale=length_scale, **kwargs
            )

        self._Kin_fn = embedded_Kin_fn
        self._Kcross_fn = embedded_Kcross_fn
        self._Kout_fn = embedded_Kout_fn

    def __call__(
        self, diffs: mm.ndarray, adjust=True, force_Kcross=False, **kwargs
    ) -> mm.ndarray:
        """
        Compute shear kernel(s) from a difference tensor.

        Args:
            diffs:
                A tensor of pairwise or crosswise distances or distances of
                shape `(data_count, nn_count, nn_count, feature_count)` or
                `(data_count, nn_count, feature_count)`.

        Returns:
            A cross-covariance matrix of shape `(data_count * 3, nn_count * 3)`
            or a tensor of shape `(data_count, nn_count * 3, nn_count * 3)`
            whose last two dimensions are kernel matrices.
        """
        if force_Kcross is True:
            return self._Kcross_fn(diffs, **kwargs)
        elif adjust and diffs.shape[-2] != diffs.shape[-3]:
            # this is probably a crosswise differences tensor
            # reshape to insert a unitary dimension
            diffs = diffs[..., None, :]
            return self._Kcross_fn(diffs, **kwargs)
        return self._Kin_fn(diffs, **kwargs)

    def Kout(self, **kwargs) -> mm.ndarray:
        return self._Kout_fn(self._backend_zeros((1, 1, 2)))

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
        return self.__call__
