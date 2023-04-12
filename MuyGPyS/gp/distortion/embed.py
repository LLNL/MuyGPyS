# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable, Union

from MuyGPyS.gp.distortion.isotropic import IsotropicDistortion
from MuyGPyS.gp.distortion.null import NullDistortion


def apply_distortion(distortion_fn: Callable, length_scale: float):
    def distortion_applier(fn: Callable):
        def distorted_fn(diffs, *args, length_scale=length_scale, **kwargs):
            return fn(distortion_fn(diffs, length_scale), *args, **kwargs)

        return distorted_fn

    return distortion_applier


def embed_with_distortion_model(
    fn: Callable, distortion_fn: Callable, length_scale: float
):
    if isinstance(distortion_fn, IsotropicDistortion):
        return apply_distortion(distortion_fn, length_scale)(fn)
    elif isinstance(distortion_fn, NullDistortion):
        return fn
    else:
        raise ValueError(f"Noise model {type(distortion_fn)} is not supported!")
