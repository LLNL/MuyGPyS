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


def _switch_on_loss_method(
    loss_method: str,
    cross_entropy_func: Callable,
    mse_func: Callable,
    lool_func: Callable,
    *args,
    **kwargs,
):
    loss_method = loss_method.lower()
    if loss_method in ["cross-entropy", "log"]:
        return cross_entropy_func(*args, **kwargs)
    elif loss_method == "mse":
        return mse_func(*args, **kwargs)
    elif loss_method == "lool":
        return lool_func(*args, **kwargs)
    else:
        raise NotImplementedError(
            f"Loss function {loss_method} is not implemented."
        )
