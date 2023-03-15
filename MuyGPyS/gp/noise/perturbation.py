# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise perturbation function wrapper
"""

from typing import Callable
from MuyGPyS.gp.noise.homoscedastic import HomoscedasticNoise


def noise_perturb(perturb_fn: Callable):
    def perturbed_version(fn):
        def fn_wrapper(K, *args, eps=0.0, **kwargs):
            return fn(perturb_fn(K, eps), *args, **kwargs)

        return fn_wrapper

    return perturbed_version


def noise_apply(fn: Callable, eps: HomoscedasticNoise):
    if not eps.fixed():

        def caller_fn(*args, **kwargs):
            return fn(*args, eps=kwargs["eps"], **kwargs)

    else:

        def caller_fn(*args, **kwargs):
            return fn(*args, eps=eps(), **kwargs)

    return caller_fn
