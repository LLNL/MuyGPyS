# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from scipy import optimize as opt

from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.objective import get_loss_func, loo_crossval


def _optim_chassis(
    synth_train,
    synth_test,
    nn_count,
    batch_size,
    kern="matern",
    hyper_dict=None,
    optim_bounds=None,
    loss_method="mse",
    verbose=False,
    nn_kwargs=None,
):
    """
    Execute an optimization pipeline.

    NOTE[bwp] this function is purely for testing purposes.
    """
    variance_mode = "diagonal"
    # kern = "matern"
    # verbose = True

    embedded_train = synth_train["input"]
    embedded_test = synth_test["input"]

    test_count = synth_test["input"].shape[0]
    train_count = synth_train["input"].shape[0]

    # Construct NN lookup datastructure.
    train_nbrs_lookup = NN_Wrapper(
        embedded_train,
        nn_count,
        **nn_kwargs,
    )
    # Make MuyGPS object
    muygps = MuyGPS(kern=kern)
    if hyper_dict is None:
        hyper_dict = dict()
    unset_params = muygps.set_params(**hyper_dict)
    do_sigma = False
    if "sigma_sq" in unset_params:
        unset_params.remove("sigma_sq")
        if variance_mode is not None:
            do_sigma = True

    if optim_bounds != None:
        muygps.set_optim_bounds(**optim_bounds)

    # Train hyperparameters by maximizing LOO predictions for batched
    # observations if `hyper_dict` unspecified.
    if len(unset_params) > 0 or do_sigma is True:
        # collect batch
        batch_indices, batch_nn_indices = sample_batch(
            train_nbrs_lookup,
            batch_size,
            train_count,
        )

    if len(unset_params) > 0:
        # set loss function
        loss_fn = get_loss_func(loss_method)

        # collect optimization settings
        bounds = muygps.optim_bounds(unset_params)
        x0 = np.array([np.random.uniform(low=b[0], high=b[1]) for b in bounds])
        if verbose is True:
            print(f"parameters to be optimized: {unset_params}")
            print(f"bounds: {bounds}")
            print(f"sampled x0: {x0}")

        # perform optimization
        optres = opt.minimize(
            loo_crossval,
            x0,
            args=(
                loss_fn,
                muygps,
                unset_params,
                batch_indices,
                batch_nn_indices,
                embedded_train,
                synth_train["output"],
            ),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if verbose is True:
            print(f"optimizer results: \n{optres}")
        muygps.set_param_array(unset_params, optres.x)
        return optres.x

    if do_sigma is True:
        muygps.sigma_sq_optim(
            batch_indices,
            batch_nn_indices,
            embedded_train,
            synth_train["output"],
        )
        return muygps.sigma_sq, muygps.get_sigma_sq(
            batch_indices,
            batch_nn_indices,
            embedded_train,
            synth_train["output"][:, 0],
        )
    #     print(f"sigma_sq results: {muygps.sigma_sq}")

    # if do_sigma is True:
    #     return muygps.get_sigma_sq(
    #         batch_indices,
    #         batch_nn_indices,
    #         embedded_train,
    #         synth_train["output"][:, 0],
    # )
