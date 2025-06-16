# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
SOAP Kernel Functor

Kernel functor for the Smooth Overlap of Atomic Positions (SOAP) 
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