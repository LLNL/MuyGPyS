# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable, Union, Dict

from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.distortion.anisotropic import AnisotropicDistortion
from MuyGPyS.gp.distortion.isotropic import IsotropicDistortion
from MuyGPyS.gp.distortion.null import NullDistortion


def apply_distortion(distortion_fn: Callable, **length_scales):
    def distortion_applier(fn: Callable):
        def distorted_fn(diffs, *args, **kwargs):
            inner_kwargs = {
                key: _optional_invoke_param(kwargs[key])
                for key in kwargs
                if key.startswith("length_scale")
            }
            for ls in length_scales:
                inner_kwargs.setdefault(
                    ls, _optional_invoke_param(length_scales[ls])
                )
            outer_kwargs = {
                key: _optional_invoke_param(kwargs[key])
                for key in kwargs
                if not key.startswith("length_scale")
            }
            return fn(
                distortion_fn(diffs, **inner_kwargs), *args, **outer_kwargs
            )

        return distorted_fn

    return distortion_applier


def _optional_invoke_param(param: Union[ScalarHyperparameter, float]) -> float:
    if isinstance(param, ScalarHyperparameter):
        return param()
    return param


def embed_with_distortion_model(
    fn: Callable,
    distortion_fn: Callable,
    length_scale: Union[ScalarHyperparameter, Dict[str, ScalarHyperparameter]],
):
    if isinstance(length_scale, ScalarHyperparameter):
        length_scale = {"length_scale": length_scale}
    if isinstance(distortion_fn, AnisotropicDistortion) or isinstance(
        distortion_fn, IsotropicDistortion
    ):
        return apply_distortion(distortion_fn, **length_scale)(fn)
    elif isinstance(distortion_fn, NullDistortion):
        return fn
    else:
        raise ValueError(f"Noise model {type(distortion_fn)} is not supported!")
