# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise perturbation function wrapper
"""

from typing import Callable

from MuyGPyS._src.gp.noise import _homoscedastic_perturb
from MuyGPyS.gp.noise.homoscedastic import HomoscedasticNoise


def noise_perturb(perturb_fn: Callable):
    def perturbed_version(fn):
        def fn_wrapper(K, *args, eps=0.0, **kwargs):
            return fn(perturb_fn(K, eps), *args, **kwargs)

        return fn_wrapper

    return perturbed_version


def perturb_with_noise_model(fn: Callable, eps: HomoscedasticNoise):
    if isinstance(eps, HomoscedasticNoise):
        return noise_perturb(_homoscedastic_perturb)(fn)
    else:
        raise ValueError(f"Noise model {type(eps)} is not supported")
