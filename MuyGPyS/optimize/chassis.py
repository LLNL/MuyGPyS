# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from scipy import optimize as opt

from MuyGPyS.optimize.objective import get_loss_func, loo_crossval


def scipy_optimize(
    muygps,
    pairwise_dists,
    crosswise_dists,
    batch_nn_indices,
    batch_indices,
    train_targets,
    loss_method="mse",
    verbose=False,
):
    loss_fn = get_loss_func(loss_method)
    optim_params = muygps.get_optim_params()
    for key in optim_params:
        optim_params[key]._set_val("sample")
    x0 = np.array([optim_params[p]() for p in optim_params])
    bounds = np.array([optim_params[p].get_bounds() for p in optim_params])
    if verbose is True:
        print(f"parameters to be optimized: {[p for p in optim_params]}")
        print(f"bounds: {bounds}")
        print(f"sampled x0: {x0}")

    batch_nn_targets = train_targets[batch_nn_indices, :]
    batch_targets = train_targets[batch_indices, :]

    optres = opt.minimize(
        loo_crossval,
        x0,
        args=(
            loss_fn,
            muygps,
            optim_params,
            pairwise_dists,
            crosswise_dists,
            batch_nn_targets,
            batch_targets,
        ),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if verbose is True:
        print(f"optimizer results: \n{optres}")

    # set final values
    for i, key in enumerate(optim_params):
        optim_params[key]._set_val(x0[i])
    return optres.x

def