# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS 
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from MuyGPyS.embed import apply_embedding
from MuyGPyS.data.utils import normalize
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.objective import (
    loo_crossval,
    get_loss_func,
)
from MuyGPyS.predict import regress_any
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.gp.muygps import MuyGPS

from scipy import optimize as opt
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from time import perf_counter


def do_regress(
    train,
    test,
    nn_count=30,
    embed_dim=30,
    batch_size=200,
    kern="matern",
    embed_method="pca",
    loss_method="mse",
    hyper_dict=None,
    nn_kwargs=None,
    optim_bounds=None,
    variance_mode=None,
    do_normalize=False,
    verbose=False,
):
    """
    Performs classification using MuyGPS.

    Parameters
    ----------
    train : dict
        A dict with keys "input" and "output". "input" maps to a matrix of row
        observation vectors. "output" maps to a matrix listing the observed
        responses of the phenomenon under study.
    test : dict
        A dict with keys "input" and "output". "input" maps to a matrix of row
        observation vectors. "output" maps to a matrix listing the observed
        responses of the phenomenon under study.
    nn_count : int
        The number of nearest neighbors to employ.
    embed_dim : int
        The dimension onto which data will be embedded.
    batch_size : int
        The batch size for hyperparameter optimization.
    kern : str
        The kernel to use. Supports ``matern'', ``rbf'', and ``nngp''.
    embed_method : str
        The embedding method to use.
        NOTE[bwp]: Currently supports only ``pca'' and None.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if ``nu''
        is not None.
        NOTE[bwp]: Currently supports only ``mse'' for regression.
    hyper_dict : dict or None
        If specified, use the given parameters for the kernel. If None, perform
        hyperparamter optimization.
    nn_kwargs : dict or None
        If not None, use the given parameters for the nearest neighbor
        algorithm specified by the "nn_method" key, which must be included if
        the user wants to use a method other than the default exact algorithm.
        Parameter names vary by method. See `MuyGPyS.neighbors.NN_Wrapper` for
        the supported methods and their parameters. If None, exact nearest
        neighbors with default settings will be used.
    optim_bounds : dict or None
        If specified, set the corresponding bounds (a 2-tuple) for each
        specified hyperparameter. If None, use all default bounds for
        hyperparameter optimization.
    variance_mode : str or None
        Specifies the type of variance to return. Currently supports
        ``diagonal'' and None. If None, report no variance term.
    do_normalize : Boolean
        Flag indicating whether to normalize. Currently redundant, but might
        want to keep this for now.
    verbose : Boolean
        If ``True'', print summary statistics.

    Returns
    -------
    predictions : numpy.ndarray(float), shape = ``(test_count, response_count)''
        The predicted response associated with each test observation.
    variance : numpy.ndarray(float), shape = ``(test_count,)''
        The unscaled independent posterior variance associated with each testing
        location. Only returned if ``variance_mode == "diagonal"''.
    sigma_sq : numpy.ndarray(float), shape = ``(response_count,)''
        The scaling parameter (for each output dimension) to be applied to the
        posterior variance. Only returned if ``variance_mode == "diagonal"''.
    """
    test_count = test["input"].shape[0]
    train_count = train["input"].shape[0]
    time_start = perf_counter()

    # Perform embedding
    embedded_train, embedded_test = apply_embedding(
        train["input"], test["input"], embed_dim, embed_method, do_normalize
    )
    time_embed = perf_counter()

    # Construct NN lookup datastructure.
    if nn_kwargs is None:
        nn_kwargs = dict()
    train_nbrs_lookup = NN_Wrapper(
        embedded_train,
        nn_count,
        **nn_kwargs,
    )
    time_nn = perf_counter()
    time_batch = perf_counter()

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
    if optim_bounds is not None:
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
                train["output"],
            ),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if verbose is True:
            print(f"optimizer results: \n{optres}")
        muygps.set_param_array(unset_params, optres.x)

    if do_sigma is True:
        muygps.sigma_sq_optim(
            batch_indices,
            batch_nn_indices,
            embedded_train,
            train["output"],
        )
        if verbose is True:
            print(f"sigma_sq results: {muygps.sigma_sq}")

    time_hyperopt = perf_counter()

    # Prediction on test data.
    predictions, pred_timing = regress_any(
        muygps,
        embedded_test,
        embedded_train,
        train_nbrs_lookup,
        train["output"],
        nn_count,
        variance_mode=variance_mode,
    )
    time_pred = perf_counter()

    # Profiling printouts if requested.
    if verbose is True:
        # record timing
        timing = {
            "embed": time_embed - time_start,
            "nn": time_nn - time_embed,
            "batch": time_batch - time_nn,
            "hyperopt": time_hyperopt - time_batch,
            "pred": time_pred - time_hyperopt,
            "pred_full": pred_timing,
        }

        print(f"muygps params : {muygps.params}")
        print(f"timing : {timing}")

    if variance_mode == "diagonal":
        predictions, variance = predictions
        return predictions, variance, muygps.sigma_sq

    return predictions
