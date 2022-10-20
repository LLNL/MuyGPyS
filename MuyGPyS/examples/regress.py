# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Resources and high-level API for a simple regression workflow.

:func:`~MuyGPyS.examples.regress.make_regressor` is a high-level API for
creating and training :class:`MuyGPyS.gp.muygps.MuyGPS` objects for regression.
:func:`~MuyGPyS.examples.regress.make_multivariate_regressor` is a high-level
API for creating and training :class:`MuyGPyS.gp.muygps.MultivariateMuyGPS`
objects for regression.

:func:`~MuyGPyS.examples.regress.do_regress` is a high-level api for executing
a simple, generic regression workflow given data. It calls the maker APIs
above and :func:`~MuyGPyS.examples.regress.regress_any`.
"""

import numpy as np

from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

from MuyGPyS.gp.distance import make_train_tensors
from MuyGPyS.optimize.chassis import optimize_from_tensors

from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.sigma_sq import (
    muygps_sigma_sq_optim,
    mmuygps_sigma_sq_optim,
)


def make_regressor(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "mse",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    sigma_method: Optional[str] = "analytic",
    k_kwargs: Dict = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    return_distances: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[MuyGPS, NN_Wrapper], Tuple[MuyGPS, NN_Wrapper, np.ndarray, np.ndarray]
]:
    """
    Convenience function for creating MuyGPyS functor and neighbor lookup data
    structure.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Example:
        >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
        >>> from MuyGPyS.examples.regress import make_regressor
        >>> train_features, train_responses = make_train()  # stand-in function
        >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
        >>> k_kwargs = {
        ...         "kern": "rbf",
        ...         "metric": "F2",
        ...         "eps": {"val": 1e-5},
        ...         "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)}
        ... }
        >>> muygps, nbrs_lookup = make_regressor(
        ...         train_features,
        ...         train_responses,
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         sigma_method="analytic",
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )
        >>> # Can alternately return distance tensors for reuse
        >>> muygps, nbrs_lookup, crosswise_dists, pairwise_dists = make_regressor(
        ...         train_features,
        ...         train_responses,
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         sigma_method="analytic",
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         return_distances=True,
        ...         verbose=False,
        ... )

    Args:
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.
        nn_count:
            The number of nearest neighbors to employ.
        batch_count:
            The number of elements to sample batch for hyperparameter
            optimization.
        loss_method:
            The loss method to use in hyperparameter optimization. Ignored if
            all of the parameters specified by argument `k_kwargs` are fixed.
            Currently supports only `"mse"` for regression.
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` and `"scipy"`.
        sigma_method:
            The optimization method to be employed to learn the `sigma_sq`
            hyperparameter. Currently supports only `"analytic"` and `None`. If
            the value is not `None`, the returned
            :class:`MuyGPyS.gp.muygps.MuyGPS` object will possess a `sigma_sq`
            member whose value, invoked via `muygps.sigma_sq()`, is a
            `(response_count,)` vector to be used for scaling posterior
            variances.
        k_kwargs:
            Parameters for the kernel, possibly including kernel type, distance
            metric, epsilon and sigma hyperparameter specifications, and
            specifications for kernel hyperparameters. See
            :ref:`MuyGPyS-gp-kernels` for examples and requirements. If all of
            the hyperparameters are fixed or are not given optimization bounds,
            no optimization will occur.
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.
        opt_kwargs:
            Parameters for the wrapped optimizer. See the docs of the
            corresponding library for supported parameters.
        return_distances:
            If `True` and any training occurs, returns a
            `(batch_count, nn_count)` matrix containing the crosswise distances
            between the batch's elements and their nearest neighbor sets and a
            `(batch_count, nn_count, nn_count)` matrix containing the pairwise
            distances between the batch's nearest neighbor sets.
        verbose:
            If `True`, print summary statistics.

    Returns
    -------
    muygps:
        A (possibly trained) MuyGPs object.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        `train_features`.
    crosswise_dists:
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
        Only returned if `return_distances is True`.
    pairwise_dists:
        A tensor of shape `(batch_count, nn_count, nn_count,)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements. Only returned if
        `return_distances is True`.
    """
    train_count, _ = train_features.shape
    _, response_count = train_targets.shape
    time_start = perf_counter()

    nbrs_lookup = NN_Wrapper(
        train_features,
        nn_count,
        **nn_kwargs,
    )
    time_nn = perf_counter()

    # create MuyGPs object
    muygps = MuyGPS(**k_kwargs)

    skip_opt = muygps.fixed()
    if skip_opt is False or sigma_method is not None:
        # collect batch
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup,
            batch_count,
            train_count,
        )
        time_batch = perf_counter()

        (
            crosswise_dists,
            pairwise_dists,
            batch_targets,
            batch_nn_targets,
        ) = make_train_tensors(
            muygps.kernel.metric,
            batch_indices,
            batch_nn_indices,
            train_features,
            train_targets,
        )

        time_tensor = perf_counter()

        if skip_opt is False:
            # maybe do something with these estimates?
            muygps = optimize_from_tensors(
                muygps,
                batch_targets,
                batch_nn_targets,
                crosswise_dists,
                pairwise_dists,
                loss_method=loss_method,
                obj_method=obj_method,
                opt_method=opt_method,
                sigma_method=sigma_method,
                verbose=verbose,
                **opt_kwargs,
            )
        time_opt = perf_counter()

        if sigma_method is not None:
            muygps = muygps_sigma_sq_optim(
                muygps,
                pairwise_dists,
                batch_nn_targets,
                sigma_method=sigma_method,
            )
            if verbose is True:
                print(f"Optimized sigma_sq values " f"{muygps.sigma_sq()}")
        time_sopt = perf_counter()

        if verbose is True:
            print(f"NN lookup creation time: {time_nn - time_start}s")
            print(f"batch sampling time: {time_batch - time_nn}s")
            print(f"tensor creation time: {time_tensor - time_batch}s")
            print(f"hyper opt time: {time_opt - time_tensor}s")
            print(f"sigma_sq opt time: {time_sopt - time_opt}s")

        if return_distances is True:
            return muygps, nbrs_lookup, crosswise_dists, pairwise_dists

    return muygps, nbrs_lookup


def make_multivariate_regressor(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "mse",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    sigma_method: Optional[str] = "analytic",
    kern: str = "matern",
    k_args: Union[List[Dict], Tuple[Dict, ...]] = list(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    return_distances: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[MMuyGPS, NN_Wrapper],
    Tuple[MMuyGPS, NN_Wrapper, np.ndarray, np.ndarray],
]:
    """
    Convenience function for creating a Multivariate MuyGPyS functor and
    neighbor lookup data structure.

    Expected parameters include a list of keyword argument dicts specifying
    kernel parameters and a dict listing nearest neighbor parameters. See the
    docstrings of the appropriate functions for specifics.

    Example:
        >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
        >>> from MuyGPyS.examples.regress import make_regressor
        >>> train_features, train_responses = make_train()  # stand-in function
        >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
        >>> k_args = [
        ...         {
        ...             "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)}
        ...             "eps": {"val": 1e-5},
        ...         },
        ...         {
        ...             "length_scale": {"val": 1.5, "bounds": (1e-2, 1e2)}
        ...             "eps": {"val": 1e-5},
        ...         },
        ... ]
        >>> mmuygps, nbrs_lookup = make_multivariate_regressor(
        ...         train_features,
        ...         train_responses,
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         sigma_method="analytic",
        ...         kern="rbf",
        ...         k_args=k_args,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )
        >>> # Can alternately return distance tensors for reuse
        >>> mmuygps, nbrs_lookup = make_multivariate_regressor(
        ...         train_features,
        ...         train_responses,
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         sigma_method="analytic",
        ...         kern="rbf",
        ...         k_args=k_args,
        ...         nn_kwargs=nn_kwargs,
        ...         return_distances=return_distances,
        ...         verbose=False,
        ... )

    Args:
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.
        nn_count:
            The number of nearest neighbors to employ.
        batch_count:
            The number of elements to sample batch for hyperparameter
            optimization.
        loss_method:
            The loss method to use in hyperparameter optimization. Ignored if
            all of the parameters specified by argument `k_kwargs` are fixed.
            Currently supports only `"mse"` for regression.
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` and `"scipy"`.
        sigma_method:
            The optimization method to be employed to learn the `sigma_sq`
            hyperparameter. Currently supports only `"analytic"` and `None`. If
            the value is not `None`, the returned
            :class:`MuyGPyS.gp.muygps.MultivariateMuyGPS` object will possess a
            `sigma_sq` member whose value, invoked via `mmuygps.sigma_sq()`, is
            a `(response_count,)` vector to be used for scaling posterior
            variances.
        kern:
            The kernel function to be used. See :ref:`MuyGPyS-gp-kernels` for
            details.
        k_args:
            A list of `response_count` dicts containing kernel initialization
            keyword arguments. Each dict specifies parameters for the kernel,
            possibly including epsilon and sigma hyperparameter specifications
            and specifications for specific kernel hyperparameters. If all of
            the hyperparameters are fixed or are not given optimization bounds,
            no optimization will occur.
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.
        opt_kwargs:
            Parameters for the wrapped optimizer. See the docs of the
            corresponding library for supported parameters.
        return_distances:
            If `True` and any training occurs, returns a
            `(batch_count, nn_count)` matrix containing the crosswise distances
            between the batch's elements and their nearest neighbor sets and a
            `(batch_count, nn_count, nn_count)` matrix containing the pairwise
            distances between the batch's nearest neighbor sets.
        verbose:
            If `True`, print summary statistics.

    Returns
    -------
    mmuygps:
        A Multivariate MuyGPs object with a separate (possibly trained) kernel
        function associated with each response dimension.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        `train_features`.
    crosswise_dists:
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
        Only returned if `return_distances is True`.
    pairwise_dists:
        A tensor of shape `(batch_count, nn_count, nn_count,)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements. Only returned if
        `return_distances is True`.
    """
    train_count, response_count = train_targets.shape
    if response_count != len(k_args):
        raise ValueError(
            f"supplied arguments for {len(k_args)} kernels, which does not "
            f"match expected {response_count} responses!"
        )
    time_start = perf_counter()

    nbrs_lookup = NN_Wrapper(
        train_features,
        nn_count,
        **nn_kwargs,
    )
    time_nn = perf_counter()

    # create MuyGPs object
    mmuygps = MMuyGPS(kern, *k_args)

    skip_opt = mmuygps.fixed()
    if skip_opt is False or sigma_method is not None:
        # collect batch
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup,
            batch_count,
            train_count,
        )
        time_batch = perf_counter()

        (
            crosswise_dists,
            pairwise_dists,
            batch_targets,
            batch_nn_targets,
        ) = make_train_tensors(
            mmuygps.metric,
            batch_indices,
            batch_nn_indices,
            train_features,
            train_targets,
        )
        time_tensor = perf_counter()

        if skip_opt is False:
            # maybe do something with these estimates?
            for i, muygps in enumerate(mmuygps.models):
                if muygps.fixed() is False:
                    mmuygps.models[i] = optimize_from_tensors(
                        muygps,
                        batch_targets[:, i].reshape(batch_targets.shape[0], 1),
                        batch_nn_targets[:, :, i].reshape(
                            batch_nn_targets.shape[0], nn_count, 1
                        ),
                        crosswise_dists,
                        pairwise_dists,
                        loss_method=loss_method,
                        obj_method=obj_method,
                        opt_method=opt_method,
                        sigma_method=sigma_method,
                        verbose=verbose,
                        **opt_kwargs,
                    )
        time_opt = perf_counter()

        if sigma_method is not None:
            mmuygps = mmuygps_sigma_sq_optim(
                mmuygps,
                pairwise_dists,
                batch_nn_targets,
                sigma_method=sigma_method,
            )
            if verbose is True:
                print(f"Optimized sigma_sq values " f"{mmuygps.sigma_sq()}")
        time_sopt = perf_counter()

        if verbose is True:
            print(f"NN lookup creation time: {time_nn - time_start}s")
            print(f"batch sampling time: {time_batch - time_nn}s")
            print(f"tensor creation time: {time_tensor - time_batch}s")
            print(f"hyper opt time: {time_opt - time_tensor}s")
            print(f"sigma_sq opt time: {time_sopt - time_opt}s")

        if return_distances is True:
            return mmuygps, nbrs_lookup, crosswise_dists, pairwise_dists

    return mmuygps, nbrs_lookup


def _empirical_covariance(train_targets: np.ndarray) -> np.ndarray:
    """
    Convenience function computing the empirical covariance kernel between
    multivariate response variables.

    Args:
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.

    Returns:
        An empirical covariance matrix of shape
        `(response_count, response_count)`
    """
    return np.cov(train_targets.T)


def _empirical_correlation(train_targets: np.ndarray) -> np.ndarray:
    """
    Convenience function computing the empirical correlation kernel between
    multivariate response variables.

    Args:
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.

    Returns:
        An empirical correlation matrix of shape
        `(response_count, response_count)`
    """
    return np.corrcoef(train_targets.T)


def _decide_and_make_regressor(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "mse",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    sigma_method: Optional[str] = "analytic",
    kern: Optional[str] = None,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    return_distances: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper],
    Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray, np.ndarray],
]:
    if kern is not None and (
        isinstance(k_kwargs, list) or isinstance(k_kwargs, tuple)
    ):
        return make_multivariate_regressor(
            train_features,
            train_targets,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            sigma_method=sigma_method,
            kern=kern,
            k_args=k_kwargs,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            return_distances=return_distances,
            verbose=verbose,
        )
    else:
        if isinstance(k_kwargs, dict):
            return make_regressor(
                train_features,
                train_targets,
                nn_count=nn_count,
                batch_count=batch_count,
                loss_method=loss_method,
                obj_method=obj_method,
                opt_method=opt_method,
                sigma_method=sigma_method,
                k_kwargs=k_kwargs,
                nn_kwargs=nn_kwargs,
                opt_kwargs=opt_kwargs,
                return_distances=return_distances,
                verbose=verbose,
            )
        else:
            raise ValueError("Expected k_kwargs to be a dict!")


def _unpack(first, *rest):
    return first, rest


def do_regress(
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "mse",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    sigma_method: Optional[str] = "analytic",
    variance_mode: Optional[str] = None,
    kern: Optional[str] = None,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    apply_sigma_sq: bool = True,
    return_distances: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray],
    Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray, np.ndarray],
    Tuple[
        Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray, np.ndarray, np.ndarray
    ],
    Tuple[
        Union[MuyGPS, MMuyGPS],
        NN_Wrapper,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
]:
    """
    Convenience function initializing a model and performing regression.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Also supports workflows relying upon multivariate models. In order to create
    a multivariate model, specify the `kern` argument and pass a list of
    hyperparameter dicts to `k_kwargs`.

    Example:
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
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         variance_mode="diagonal",
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )
        >>> # Can alternately return distance tensors for reuse
        >>> muygps, nbrs_lookup, predictions, variance, crosswise_dists, pairwise_dists = do_regress(
        ...         test['input'],
        ...         train['input'],
        ...         train['output'],
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         variance_mode="diagonal",
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         return_distances=True,
        ...         verbose=False,
        ... )
        >>> mse = mse_fn(test['output'], predictions)
        >>> print(f"obtained mse: {mse}")
        obtained mse: 0.20842...

    Args:
        test_features:
            A matrix of shape `(test_count, feature_count)` whose rows consist
            of observation vectors of the test data.
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.
        nn_count:
            The number of nearest neighbors to employ.
        batch_count:
            The number of elements to sample batch for hyperparameter
            optimization.
        loss_method:
            The loss method to use in hyperparameter optimization. Ignored if
            all of the parameters specified by argument `k_kwargs` are fixed.
            Currently supports only `"mse"` for regression.
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` and `"scipy"`.
        sigma_method:
            The optimization method to be employed to learn the `sigma_sq`
            hyperparameter. Currently supports only `"analytic"` and `None`. If
            the value is not `None`, the returned
            :class:`MuyGPyS.gp.muygps.MuyGPS` object will possess a `sigma_sq`
            member whose value, invoked via `muygps.sigma_sq()`, is a
            `(response_count,)` vector to be used for scaling posterior
            variances.
        variance_mode:
            Specifies the type of variance to return. Currently supports
            `diagonal` and None. If None, report no variance term.
        kern:
            The kernel function to be used. See :ref:`MuyGPyS-gp-kernels` for
            details. Only used in the multivariate case. If `None`, assume
            that we are not using a multivariate model.
        k_kwargs:
            If given a list or tuple of length `response_count`, assume that the
            elements are dicts containing kernel initialization keyword
            arguments for the creation of a multivariate model (see
            :func:`~MuyGPyS.examples.regress.make_multivariate_regressor`).
            If given a dict, assume that the elements are keyword arguments to
            a MuyGPs model (see
            :func:`~MuyGPyS.examples.regress.make_regressor`).
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.
        opt_kwargs:
            Parameters for the wrapped optimizer. See the docs of the
            corresponding library for supported parameters.
        apply_sigma_sq:
            If `True` and `variance_mode is not None`, automatically scale the
            posterior variances by `sigma_sq`.
        return_distances:
            If `True`, returns a `(test_count, nn_count)` matrix containing the
            crosswise distances between the test elements and their nearest
            neighbor sets and a `(test_count, nn_count, nn_count)` tensor
            containing the pairwise distances between the test's nearest
            neighbor sets.
        verbose:
            If `True`, print summary statistics.

    Returns
    -------
    muygps:
        A (possibly trained) MuyGPs object.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        `train_features`.
    predictions:
        The predicted response associated with each test observation.
    variance:
        Estimated posterior variance of each test prediction. If
        `variance_mode == "diagonal"` return a `(test_count, response_count)`
        matrix where each row is the posterior variance. If
        `sigma_method is not None` and `apply_sigma_sq is True`, each column
        of the variance is automatically scaled by the corresponding `sigma_sq`
        parameter.
    crosswise_dists:
        A matrix of shape `(test_count, nn_count)` whose rows list the distance
        of the corresponding test element to each of its nearest neighbors.
        Only returned if `return_distances is True`.
    pairwise_dists:
        A tensor of shape `(test_count, nn_count, nn_count,)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the test elements. Only returned if
        `return_distances is True`.
    """
    if sigma_method is None:
        apply_sigma_sq = False

    regressor_args = _decide_and_make_regressor(
        train_features,
        train_targets,
        nn_count=nn_count,
        batch_count=batch_count,
        loss_method=loss_method,
        obj_method=obj_method,
        opt_method=opt_method,
        sigma_method=sigma_method,
        kern=kern,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        opt_kwargs=opt_kwargs,
        return_distances=False,
        verbose=verbose,
    )
    regressor, regressor_args_less1 = _unpack(*regressor_args)
    nbrs_lookup, regressor_args_less2 = _unpack(*regressor_args_less1)
    if len(regressor_args_less2) > 0:
        # Should not get here
        # crosswise_dists, pairwise_dists = regressor_args_less2
        pass

    prediction_args, pred_timing = regress_any(
        regressor,
        test_features,
        train_features,
        nbrs_lookup,
        train_targets,
        variance_mode=variance_mode,
        apply_sigma_sq=apply_sigma_sq,
        return_distances=return_distances,
    )

    # predictions, prediction_args_less1 = _unpack(*prediction_args)
    if verbose is True:
        print("prediction time breakdown:")
        for k in pred_timing:
            print(f"\t{k} time:{pred_timing[k]}s")

    if variance_mode is None and len(regressor_args_less2) == 0:
        return regressor, nbrs_lookup, prediction_args
    elif variance_mode is not None and return_distances is False:
        predictions, prediction_args_less1 = _unpack(*prediction_args)
        variance, prediction_args_less2 = _unpack(*prediction_args_less1)
        return regressor, nbrs_lookup, predictions, variance
    elif variance_mode is None and len(regressor_args_less2) > 0:
        predictions, prediction_args_less1 = _unpack(*prediction_args)
        crosswise_dists, prediction_args_less2 = _unpack(*prediction_args_less1)
        pairwise_dists, prediction_args_less3 = _unpack(*prediction_args_less2)
        return (
            regressor,
            nbrs_lookup,
            predictions,
            crosswise_dists,
            pairwise_dists,
        )
    else:
        predictions, prediction_args_less1 = _unpack(*prediction_args)
        variance, prediction_args_less2 = _unpack(*prediction_args_less1)
        crosswise_dists, prediction_args_less3 = _unpack(*prediction_args_less2)
        pairwise_dists, prediction_args_less4 = _unpack(*prediction_args_less3)
        return (
            regressor,
            nbrs_lookup,
            predictions,
            variance,
            crosswise_dists,
            pairwise_dists,
        )


def regress_any(
    regressor: Union[MuyGPS, MMuyGPS],
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_nbrs_lookup: NN_Wrapper,
    train_targets: np.ndarray,
    variance_mode: Optional[str] = None,
    apply_sigma_sq: bool = True,
    return_distances: bool = False,
) -> Union[
    Tuple[np.ndarray, Dict[str, float]],
    Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, float]],
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]],
    Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str, float]
    ],
]:
    """
    Simultaneously predicts the response for each test item.

    Args:
        regressor:
            Regressor object.
        test_features:
            Test observations of shape `(test_count, feature_count)`.
        train_features:
            Train observations of shape `(train_count, feature_count)`.
        train_nbrs_lookup:
            Trained nearest neighbor query data structure.
        train_targets:
            Observed response for all training data of shape
            `(train_count, class_count)`.
        variance_mode : str or None
            Specifies the type of variance to return. Currently supports
            `diagonal` and None. If None, report no variance term.
        apply_sigma_sq:
            If `True` and `variance_mode is not None`, automatically scale the
            posterior variances by `sigma_sq`.
        return_distances:
            If `True`, returns a `(test_count, nn_count)` matrix containing the
            crosswise distances between the test elements and their nearest
            neighbor sets and a `(test_count, nn_count, nn_count)` tensor
            containing the pairwise distances between the test data's nearest
            neighbor sets.

    Returns
    -------
    means:
        The predicted response of shape `(test_count, response_count,)` for
        each of the test examples.
    variances:
        The independent posterior variances for each of the test examples. Of
        shape `(test_count,)` if the argument `regressor` is an instance of
        :class:`MuyGPyS.gp.muygps.MuyGPS`, and of shape
        `(test_count, response_count)` if `regressor` is an instance of
        :class:`MuyGPyS.gp.muygps.MultivariateMuyGPS`. Returned only when
        `variance_mode == "diagonal"`.
    crosswise_dists:
        A matrix of shape `(test_count, nn_count)` whose rows list the distance
        of the corresponding test element to each of its nearest neighbors.
        Only returned if `return_distances is True`.
    pairwise_dists:
        A tensor of shape `(test_count, nn_count, nn_count,)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the test elements. Only returned if
        `return_distances is True`.
    timing : dict
        Timing for the subroutines of this function.
    """
    test_count, _ = test_features.shape
    # train_count, _ = train_features.shape

    time_start = perf_counter()
    test_nn_indices, _ = train_nbrs_lookup.get_nns(test_features)
    time_nn = perf_counter()

    time_agree = perf_counter()

    predictions = regressor.regress_from_indices(
        np.arange(test_count),
        test_nn_indices,
        test_features,
        train_features,
        train_targets,
        variance_mode=variance_mode,
        apply_sigma_sq=apply_sigma_sq,
        return_distances=return_distances,
    )
    time_pred = perf_counter()

    timing = {
        "nn": time_nn - time_start,
        "agree": time_agree - time_nn,
        "pred": time_pred - time_agree,
    }
    return predictions, timing
