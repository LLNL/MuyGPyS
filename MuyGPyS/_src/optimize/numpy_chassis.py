# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
from typing import Callable

from scipy import optimize as opt

from MuyGPyS.gp.muygps import MuyGPS


def _scipy_optimize_from_tensors(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
) -> MuyGPS:
    x0_names, x0, bounds = muygps.get_optim_params()
    if verbose is True:
        print(f"parameters to be optimized: {x0_names}")
        print(f"bounds: {bounds}")
        print(f"initial x0: {x0}")

    optres = opt.minimize(
        obj_fn,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )
    if verbose is True:
        print(f"optimizer results: \n{optres}")

    ret = deepcopy(muygps)

    # set final values
    for i, key in enumerate(x0_names):
        lb, ub = bounds[i]
        if optres.x[i] < lb:
            val = lb
        elif optres.x[i] > ub:
            val = ub
        else:
            val = optres.x[i]
        if key == "eps":
            ret.eps._set_val(val)
        else:
            ret.kernel.hyperparameters[key]._set_val(val)

    return ret
