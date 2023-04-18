# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable, Union, Dict

from MuyGPyS.gp.kernels import Hyperparameter
from MuyGPyS.gp.distortion.anisotropic import AnisotropicDistortion
from MuyGPyS.gp.distortion.isotropic import IsotropicDistortion
from MuyGPyS.gp.distortion.null import NullDistortion


def apply_distortion(distortion_fn: Callable, length_scale: float):
    def distortion_applier(fn: Callable):
        def distorted_fn(diffs, *args, length_scale=length_scale, **kwargs):
            return fn(distortion_fn(diffs, length_scale), *args, **kwargs)

        return distorted_fn

    return distortion_applier


def apply_anisotropic_distortion(distortion_fn: Callable, **kwargs):
    def distortion_applier(fn: Callable):
        def distorted_fn(diffs, *args, **kwargs):
            return fn(
                distortion_fn(diffs, **convert_length_scales(kwargs)),
                *args,
                **kwargs,
            )

        return distorted_fn

    return distortion_applier


def convert_length_scales(length_scales: Dict[str, Hyperparameter]):
    for key, value in length_scales.items():
        length_scales[key] = value
    return length_scales


def embed_with_distortion_model(
    fn: Callable,
    distortion_fn: Callable,
    length_scale: Union[Hyperparameter, Dict[str, Hyperparameter]],
    **kwargs,
):
    if isinstance(distortion_fn, AnisotropicDistortion):
        return apply_anisotropic_distortion(distortion_fn, **kwargs)(fn)
    if isinstance(distortion_fn, IsotropicDistortion):
        return apply_distortion(distortion_fn, length_scale())(fn)
    elif isinstance(distortion_fn, NullDistortion):
        return fn
    else:
        raise ValueError(f"Noise model {type(distortion_fn)} is not supported!")
