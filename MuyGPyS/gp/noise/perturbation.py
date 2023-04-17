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


def select_perturb_fn(
    eps: Union[HomoscedasticNoise, HeteroscedasticNoise, NullNoise]
) -> Callable:
    if isinstance(eps, HomoscedasticNoise):
        return _homoscedastic_perturb
    elif isinstance(eps, HeteroscedasticNoise):
        return _heteroscedastic_perturb
    elif isinstance(eps, NullNoise):

        def _null_noise(K, *args, **kwargs):
            return K

        return _null_noise

    else:
        raise ValueError(f"Noise model {type(eps)} is not supported")


def noise_perturb(perturb_fn: Callable):
    def perturbed_version(fn):
        def fn_wrapper(K, *args, eps=0.0, **kwargs):
            return fn(perturb_fn(K, eps), *args, **kwargs)

        return fn_wrapper

    return perturbed_version


def perturb_with_noise_model(
    fn: Callable,
    eps: Union[HomoscedasticNoise, HeteroscedasticNoise, NullNoise],
) -> Callable:
    return noise_perturb(select_perturb_fn(eps))(fn)
