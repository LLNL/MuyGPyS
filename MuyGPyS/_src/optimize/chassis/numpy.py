# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable

from bayes_opt import BayesianOptimization
from scipy import optimize as opt

from MuyGPyS.gp import MuyGPS


def _new_muygps(mugps: MuyGPS, x0_names, bounds, opt_dict) -> MuyGPS:
    ret = MuyGPS()
    for i, key in enumerate(x0_names):
        lb, ub = bounds[i]
        val = opt_dict[key]
        if val < lb:
            val = lb
        elif val > ub:
            val = ub
        if key == "eps":
            ret.eps._set_val(val)
        else:
            ret.kernel.hyperparameters[key]._set_val(val)
    return ret


def _obj_fn_adapter(obj_fn, x0_names):
    def array_obj_fn(x_array):
        arr_dict = {h: x_array[i] for i, h in enumerate(x0_names)}
        return -obj_fn(**arr_dict)

    return array_obj_fn


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

    # converting from kwargs representation to array representation
    array_obj_fn = _obj_fn_adapter(obj_fn, x0_names)

    optres = opt.minimize(
        array_obj_fn,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        **kwargs,
    )
    if verbose is True:
        print(f"optimizer results: \n{optres}")

    # converting back to kwargs representation
    ret_dict = {n: optres.x[i] for i, n in enumerate(x0_names)}

    return _new_muygps(muygps, x0_names, bounds, ret_dict)


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
            "allow_duplicate_points",
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

    return _new_muygps(muygps, x0_names, bounds, optimizer.max["params"])
