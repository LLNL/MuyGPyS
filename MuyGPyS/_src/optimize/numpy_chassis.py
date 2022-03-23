# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
from typing import Callable

from bayes_opt import BayesianOptimization
from scipy import optimize as opt

from MuyGPyS.gp.muygps import MuyGPS


def _scipy_optimize(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
    **kwargs,
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
        **kwargs,
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


def _bayes_opt_optimize(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    x0_names, x0, bounds = muygps.get_optim_params()
    if verbose is True:
        print(f"parameters to be optimized: {x0_names}")
        print(f"bounds: {bounds}")
        print(f"initial x0: {x0}")

    bounds_map = {n: bounds[i] for i, n in enumerate(x0_names)}
    x0_map = {n: x0[i] for i, n in enumerate(x0_names)}

    optimizer_kwargs = {
        k: kwargs[k]
        for k in kwargs
        if k
        in {
            "random_state",
            "verbose",
            "bounds_transformer",
        }
    }
    if "verbose" not in optimizer_kwargs:
        optimizer_kwargs["verbose"] = 2 if verbose is True else 0

    # not allowing the user to set the opt gp parameters for now
    maximize_kwargs = {
        k: kwargs[k]
        for k in kwargs
        if k
        in {
            "init_points",
            "n_iter",
            "acq",
            "kappa",
            "kappa_decay",
            "kappa_decay_delay",
            "xi",
        }
    }
    # set defaults
    if "init_points" not in maximize_kwargs:
        maximize_kwargs["init_points"] = 5
    if "n_iter" not in maximize_kwargs:
        maximize_kwargs["n_iter"] = 20

    optimizer = BayesianOptimization(
        f=obj_fn,
        pbounds=bounds_map,
        **optimizer_kwargs,
    )
    optimizer.probe(x0_map, lazy=True)
    optimizer.maximize(**maximize_kwargs)

    ret = deepcopy(muygps)

    # set final values
    for i, key in enumerate(x0_names):
        lb, ub = bounds[i]
        val = optimizer.max["params"][key]
        if val < lb:
            val = lb
        elif val > ub:
            val = ub
        if key == "eps":
            ret.eps._set_val(val)
        else:
            ret.kernel.hyperparameters[key]._set_val(val)

    return ret
