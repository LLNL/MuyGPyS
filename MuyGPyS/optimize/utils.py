# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable, Optional


def _switch_on_opt_method(
    opt_method: str, bayes_func: Callable, scipy_func: Callable, *args, **kwargs
):
    opt_method = opt_method.lower()
    if opt_method in ["bayesian", "bayes", "bayes-opt"]:
        return bayes_func(*args, **kwargs)
    elif opt_method == "scipy":
        return scipy_func(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported optimization method: {opt_method}")


def _switch_on_sigma_method(
    sigma_method: Optional[str],
    analytic_func: Callable,
    none_func: Callable,
    *args,
    **kwargs,
):
    if sigma_method is not None:
        sigma_method = sigma_method.lower()
    if sigma_method == "analytic":
        return analytic_func(*args, **kwargs)
    elif sigma_method is None:
        return none_func(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognized sigma_method {sigma_method}")
