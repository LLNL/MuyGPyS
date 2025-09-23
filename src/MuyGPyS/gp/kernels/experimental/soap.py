# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
SOAP Kernel Functor

Kernel functor for the Smooth Overlap of Atomic Positions (SOAP).
Define some of the specifics and give a bit of background.
"""
from typing import Callable, List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.kernels.soap import _soap_fn
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation import (
    DeformationFn,
    DifferenceIsotropy,
    dot,
)
from MuyGPyS.gp.kernels import KernelFn
from MuyGPyS.gp.hyperparameter import ScalarParam, NamedParam


@auto_str
class SOAPKernel(KernelFn):
    """
    The SOAP Kernel.

    Better description goes here. A lot will go into this soon, but leave bare
    bones for now.
    """

    def __init__(
        self,
        sensitivity: ScalarParam = ScalarParam(2.0),
        deformation: DeformationFn = DifferenceIsotropy(
            metric=dot, length_scale=ScalarParam(1.0)
        ),
        _backend_fn: Callable = _soap_fn,
        _backend_zeros: Callable = mm.zeros,
        _backend_squeeze: Callable = mm.squeeze,
    ):
        super().__init__(deformation=deformation)
        if not isinstance(self.deformation, DifferenceIsotropy):
            raise ValueError(
                "SOAPKernel must be an instance of DifferenceIsotropy"
                f" not {type(deformation)}"
            )
        self.sensitivity = NamedParam("sensitivity", sensitivity)
        self._kernel_fn = _backend_fn
        self._backend_zeros = _backend_zeros
        self._backend_squeeze = _backend_squeeze

        self._make()

    def _make(self):
        super()._make_base()
        self.sensitivity.populate(self._hyperparameters)

        # Need length_scale passthrough
        def embedded_fn(diffs, *args, sensitivity=None, **kwargs):
            if sensitivity is None:
                sensitivity = self.sensitivity()
            return self._kernel_fn(
                diffs, *args, sensitivity=sensitivity, **kwargs
            )

        self._fn = embedded_fn

    def __call__(self, diffs, **kwargs):
        """
        Compute the SOAP Kernel(s) from distance tensors
        """
        return self._fn(diffs, **kwargs)

    def apply_Kout_fn(self, **kwargs) -> Callable:
        def apply_Kout_fn(fn: Callable) -> Callable:
            def fixed_Kout_fn(Kin, Kcross, set_Kout, *args, **kwargs):
                return fn(Kin, Kcross, set_Kout, *args, **kwargs)

            return fixed_Kout_fn

        return apply_Kout_fn

    def get_opt_params(
        self,
    ) -> Tuple[List[str], List[float], List[Tuple[float, float]]]:
        """
        Return list of hyperparameter names, values, and bounds.
        """
        names, params, bounds = super().get_opt_params()
        self.sensitivity.append_lists(names, params, bounds)
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
