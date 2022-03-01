# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable

from MuyGPyS import config

from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS._src.optimize.numpy_chassis import (
    _scipy_optimize_from_tensors as _numpy_scipy_optimize_from_tensors,
)


def _scipy_optimize_from_tensors(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
) -> MuyGPS:
    """
    NOTE[bwp] This is presently required because scipy.optimize.minimize() does
    not work with 32-bit functions, and so we must temporarily promote jax
    functions if we are operating in 32-bit mode. We will hopefully remove this
    in a future update.
    """
    if config.x64_enabled() is False:
        config.jax_enable_x64()
        ret = _numpy_scipy_optimize_from_tensors(
            muygps, obj_fn, verbose=verbose
        )
        config.jax_disable_x64()
        return ret
    else:
        return _numpy_scipy_optimize_from_tensors(
            muygps, obj_fn, verbose=verbose
        )
