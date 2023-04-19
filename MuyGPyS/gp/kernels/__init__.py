# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from .kernel_fn import KernelFn
from .matern import Matern
from .rbf import RBF


def _get_kernel(kern: str, **kwargs) -> KernelFn:
    """
    Select and return an appropriate kernel functor based upon the passed
    parameters.

    Args:
        kern:
            The kernel function to be used. Current supports only `"matern"` and
            `"rbf"`.
        kwargs : dict
            Kernel parameters, possibly including hyperparameter dicts.

    Return:
        The appropriately initialized kernel functor.
    """
    if kern == "rbf":
        return RBF(**kwargs)
    elif kern == "matern":
        return Matern(**kwargs)
    else:
        raise ValueError(f"Kernel type {kern} is not supported!")
