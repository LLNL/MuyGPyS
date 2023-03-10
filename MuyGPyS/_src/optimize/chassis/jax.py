# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable

from MuyGPyS.gp import MuyGPS
from MuyGPyS._src.optimize.chassis.numpy import (
    _scipy_optimize,
    _bayes_opt_optimize as _numpy_bayes_opt_optimize,
)


def _bayes_opt_optimize(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    return _numpy_bayes_opt_optimize(muygps, obj_fn, verbose=verbose, **kwargs)
