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
from MuyGPyS._src.gp.kernels.soap import (
    _soap_fn
)
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation import (
    DifferenceIsotropy,
    dot,
)
from MuyGPyS.gp.kernels import KernelFn
from MuyGPyS.gp.hyperparameter import ScalarParam


@auto_str
class SOAPKernel(KernelFn):
    """
    The SOAP Kernel.

    Better description goes here. A lot will go into this soon, but leave bare bones for now.
    """

    def __init__(
            self,
            deformation: DifferenceIsotropy = DifferenceIsotropy(
                dot, length_scale=ScalarParam(1.0)
            ),
            _backend_fn: Callable = _soap_fn,
            _backend_zeros: Callable = mm.zeros
    ):
        super().__init__(deformation=deformation)
        if not isinstance(self.deformation, DifferenceIsotropy):
            raise ValueError(
                "SOAPKernel must be an instance of DifferenceIsotropy"
                f" not {type(deformation)}"
            )
        self._kernel_fn = _backend_fn
        self._backend_zeros = _backend_zeros
        self._make()

    def _make(self):
        super()._make_base()

        # do the ls passthrough like in the shaer kernel
        # helps because current implementation 
        def embedded_fn(diffs, *args, length_scale=None, **kwargs):
            if length_scale is None:
                length_scale = self.deformation.length_scale()
            return self._kernel_fn(
                diffs, *args, length_scale=length_scale, **kwargs
            )
        
        self._fn = embedded_fn

    def __call__(self, diffs: mm.ndarray, adjust=True, **kwargs) -> mm.ndarray:
        """
        Compute the SOAP Kernel(s) from distance tensors
        """
        if adjust and diffs.shape[-4] != diffs.shape[-6]:
            # add unitary dimension to crosswise tensor
            diffs = diffs[..., None, :, :, :]

            return self._fn(diffs, **kwargs)
        
    def Kout(self, **kwargs) -> mm.ndarray:
        return self.__call__(self._backend_zeros((1, 1, 1, 1, 1, 1, 1)))
        
    def get_opt_params(
            self,
    ) -> Tuple[List[str], List[float], List[Tuple[float, float]]]:
        """
        Return list of hyperparameter names, values, and bounds.
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
