# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable

from MuyGPyS import config, jax_config

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
    if config.muygpys_jax_enabled is True:  # type: ignore
        if jax_config.x64_enabled is False:  # type: ignore
            jax_config.update("jax_enable_x64", True)
            ret = _numpy_scipy_optimize_from_tensors(
                muygps, obj_fn, verbose=verbose
            )
            jax_config.update("jax_enable_x64", False)
            return ret
    return _numpy_scipy_optimize_from_tensors(muygps, obj_fn, verbose=verbose)
