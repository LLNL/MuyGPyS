# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise perturbation function wrapper
"""

from typing import Callable, Union

from MuyGPyS._src.gp.noise import (
    _homoscedastic_perturb,
    _heteroscedastic_perturb,
)
from MuyGPyS.gp.noise.homoscedastic import HomoscedasticNoise
from MuyGPyS.gp.noise.heteroscedastic import HeteroscedasticNoise
from MuyGPyS.gp.noise.null import NullNoise


def noise_perturb(perturb_fn: Callable):
    def perturbed_version(fn):
        def fn_wrapper(K, *args, eps=0.0, **kwargs):
            return fn(perturb_fn(K, eps), *args, **kwargs)

        return fn_wrapper

    return perturbed_version


def perturb_with_noise_model(
    fn: Callable,
    eps: Union[HomoscedasticNoise, HeteroscedasticNoise, NullNoise],
):
    if isinstance(eps, HomoscedasticNoise):
        return noise_perturb(_homoscedastic_perturb)(fn)
    elif isinstance(eps, HeteroscedasticNoise):
        return noise_perturb(_heteroscedastic_perturb)(fn)
    elif isinstance(eps, NullNoise):
        return fn
    else:
        raise ValueError(f"Noise model {type(eps)} is not supported")
