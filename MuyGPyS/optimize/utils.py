# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable


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
