# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from MuyGPyS.optimize.chassis import scipy_optimize_from_tensors
from MuyGPyS.gp.distance import crosswise_distances, pairwise_distances

# import numpy as np

# from MuyGPyS.embed import apply_embedding
# from MuyGPyS.data.utils import normalize
from MuyGPyS.optimize.batch import sample_batch

# from MuyGPyS.optimize.objective import (
#     loo_crossval,
#     get_loss_func,
# )
from MuyGPyS.predict import regress_any
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.gp.muygps import MuyGPS

# from scipy import optimize as opt
from time import perf_counter


def make_regressor(
    train_data,
    train_targets,
    nn_count=30,
    batch_size=200,
    loss_method="mse",
    k_kwargs=dict(),
    nn_kwargs=dict(),
    verbose=False,
):
    """
    Convenience function for creating MuyGPyS functor and neighbor lookup data
    structure.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Parameters
    ----------
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors.
    train_targets = numpy.ndarray(float),
                    shape = ``(train_count, response_count)''
        A matrix of row response vectors for the training data.
    nn_count : int
        The number of nearest neighbors to employ.
    batch_size : int
        The batch size for hyperparameter optimization.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if all of
        the parameters specified by ``k_kwargs'' are fixed.
        NOTE[bwp]: Currently supports only ``mse'' for regression.
    k_kwargs : dict
        Parameters for the kernel, possibly including kernel type, distance
        metric, epsilon and sigma hyperparameter specifications, and
        specifications for kernel hyperparameters. If all of the hyperparameters
        are fixed or are not given optimization bounds, no optimization will
        occur.
    nn_kwargs : dict
        Parameters for the nearest neighbors wrapper. See
        `MuyGPyS.neighbors.NN_Wrapper` for the supported methods and their
        parameters.
    verbose : Boolean
        If ``True'', print summary statistics.

    Returns
    -------
    muygps : MuyGPyS.gp.MuyGPS
        A (possibly trained) MuyGPs object.
    nbrs_lookup : MuyGPyS.neighbors.NN_Wrapper
        A data structure supporting nearest neighbor queries into
        ``train_data''.

    Examples
    --------
    >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
    >>> from MuyGPyS.examples.regress import make_regressor
    >>> train = _make_gaussian_dict(10000, 100, 10)
    >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
    >>> k_kwargs = {
    ...         "kern": "rbf",
    ...         "metric": "F2",
    ...         "eps": {"val": 1e-5},
    ...         "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)}
    ... }
    >>> muygps, nbrs_lookup = make_regressor(
    ...         train['input'],
    ...         train['output'],
    ...         nn_count=30,
    ...         batch_size=200,
    ...         loss_method="mse",
    ...         k_kwargs=k_kwargs,
    ...         nn_kwargs=nn_kwargs,
    ...         verbose=False,
    ... )
    """
    train_count, _ = train_data.shape
    time_start = perf_counter()

    nbrs_lookup = NN_Wrapper(
        train_data,
        nn_count,
        **nn_kwargs,
    )
    time_nn = perf_counter()

    muygps = MuyGPS(**k_kwargs)
    skip_opt = muygps.fixed_nosigmasq()
    skip_sigma = muygps.fixed_sigmasq()
    if skip_opt is False or skip_sigma is False:
        # collect batch
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup,
            batch_size,
            train_count,
        )
        time_batch = perf_counter()

        crosswise_dists = crosswise_distances(
            train_data,
            train_data,
            batch_indices,
            batch_nn_indices,
            metric=muygps.kernel.metric,
        )
        pairwise_dists = pairwise_distances(
            train_data, batch_nn_indices, metric=muygps.kernel.metric
        )
        time_tensor = perf_counter()

        if skip_opt is False:
            # maybe do something with these estimates?
            estimates = scipy_optimize_from_tensors(
                muygps,
                batch_indices,
                batch_nn_indices,
                crosswise_dists,
                pairwise_dists,
                train_targets,
                loss_method=loss_method,
                verbose=verbose,
            )
        time_opt = perf_counter()

        if skip_sigma is False:
            K = muygps.kernel(pairwise_dists)
            muygps.sigma_sq_optim(K, batch_nn_indices, train_targets)
        time_sopt = perf_counter()

        if verbose is True:
            print(f"NN lookup creation time: {time_nn - time_start}s")
            print(f"batch sampling time: {time_batch - time_nn}s")
            print(f"tensor creation time: {time_tensor - time_batch}s")
            print(f"hyper opt time: {time_opt - time_tensor}s")
            print(f"sigma_sq opt time: {time_sopt - time_opt}s")

    return muygps, nbrs_lookup


def do_regress(
    test_data,
    train_data,
    train_targets,
    nn_count=30,
    batch_size=200,
    loss_method="mse",
    variance_mode=None,
    k_kwargs=dict(),
    nn_kwargs=dict(),
    verbose=False,
):
    """
    Convenience function initializing a model and performing regression.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Parameters
    ----------
    test_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the test data.
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the train data.
    train_targets = numpy.ndarray(float),
                    shape = ``(train_count, response_count)''
        A matrix of row response vectors of the train data.
    nn_count : int
        The number of nearest neighbors to employ.
    batch_size : int
        The batch size for hyperparameter optimization.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if all of
        the parameters specified by ``k_kwargs'' are fixed.
        NOTE[bwp]: Currently supports only ``mse'' for regression.
    variance_mode : str or None
        Specifies the type of variance to return. Currently supports
        ``diagonal'' and None. If None, report no variance term.
    k_kwargs : dict
        Parameters for the kernel, possibly including kernel type, distance
        metric, epsilon and sigma hyperparameter specifications, and
        specifications for kernel hyperparameters. If all of the hyperparameters
        are fixed or are not given optimization bounds, no optimization will
        occur.
    nn_kwargs : dict
        Parameters for the nearest neighbors wrapper. See
        `MuyGPyS.neighbors.NN_Wrapper` for the supported methods and their
        parameters.
    verbose : Boolean
        If ``True'', print summary statistics.

    Returns
    -------
    muygps : MuyGPyS.gp.MuyGPS
        A (possibly trained) MuyGPs object.
    nbrs_lookup : MuyGPyS.neighbors.NN_Wrapper
        A data structure supporting nearest neighbor queries into
        ``train_data''.
    predictions : numpy.ndarray(float), shape = ``(test_count, response_count)''
        The predicted response associated with each test observation.
    variance : numpy.ndarray(float), shape = ``(test_count,)''
        Estimated posterior variance of each test prediction. Only returned if
        ``variance_mode'' is not None.

    Examples
    --------
    >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
    >>> from MuyGPyS.examples.regress import do_regress
    >>> from MuyGPyS.optimize.objective import mse_fn
    >>> train, test = _make_gaussian_data(10000, 1000, 100, 10)
    >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
    >>> k_kwargs = {
    ...         "kern": "rbf",
    ...         "metric": "F2",
    ...         "eps": {"val": 1e-5},
    ...         "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)}
    ... }
    >>> muygps, nbrs_lookup, predictions, variance = do_regress(
    ...         test['input'],
    ...         train['input'],
    ...         train['output'],
    ...         nn_count=30,
    ...         batch_size=200,
    ...         loss_method="mse",
    ...         variance_mode="diagonal",
    ...         k_kwargs=k_kwargs,
    ...         nn_kwargs=nn_kwargs,
    ...         verbose=False,
    ... )
    >>> mse = mse_fn(test['output'], predictions)
    >>> print(f"obtained mse: {mse}")
    obtained mse: 0.20842...
    """

    muygps, nbrs_lookup = make_regressor(
        train_data,
        train_targets,
        nn_count=nn_count,
        batch_size=batch_size,
        loss_method=loss_method,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        verbose=verbose,
    )

    predictions, pred_timing = regress_any(
        muygps,
        test_data,
        train_data,
        nbrs_lookup,
        train_targets,
        variance_mode=variance_mode,
    )
    if verbose is True:
        print(f"prediction time breakdown:")
        for k in pred_timing:
            print(f"\t{k} time:{pred_timing[k]}s")
    return muygps, nbrs_lookup, predictions
