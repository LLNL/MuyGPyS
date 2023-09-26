# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Resources and high-level API for a fast posterior mean inference workflow.

:func:`~MuyGPyS.examples.fast_posterior_mean.make_fast_regressor` is a
high-level API for creating the necessary components for fast posterior mean
inference.
:func:`~MuyGPyS.examples.fast_posterior_mean.make_fast_multivariate_regressor`
is a high-level API for creating the necessary components for fast posterior
mean inference with multiple outputs.

:func:`~MuyGPyS.examples.fast_posterior_mean.do_fast_posterior_mean` is a
high-level api for executing a simple, generic fast posterior medan workflow
given data.
It calls the maker APIs above and
:func:`~MuyGPyS.examples.fast_posterior_mean.fast_posterior_mean_any`.
"""

from time import perf_counter
from typing import Dict, List, Tuple, Union

import MuyGPyS._src.math as mm
from MuyGPyS.gp import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.examples.from_indices import fast_posterior_mean_from_indices
from MuyGPyS.examples.regress import _decide_and_make_regressor
from MuyGPyS.gp.tensors import fast_nn_update
from MuyGPyS.gp.tensors import pairwise_tensor
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import Bayes_optimize, OptimizeFn
from MuyGPyS.optimize.loss import LossFn, lool_fn


def make_fast_regressor(
    muygps: MuyGPS,
    nbrs_lookup: NN_Wrapper,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Convenience function for creating precomputed coefficient matrix and neighbor lookup data
    structure.

    Args:
        muygps:
            A (possibly trained) MuyGPS object.
        nbrs_lookup:
             A data structure supporting nearest neighbor queries into
            `train_features`.
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.

    Returns
    -------
    precomputed_coefficients_matrix:
        A matrix of shape `(train_count, nn_count)` whose rows list the
        precomputed coefficients for each nearest neighbors set in the
        training data.
    nn_indices:
        A numpy.ndarrray supporting nearest neighbor queries.
    """

    num_training_samples, _ = train_features.shape
    nn_indices, _ = nbrs_lookup.get_batch_nns(
        mm.arange(0, num_training_samples)
    )
    nn_indices = fast_nn_update(nn_indices)

    train_nn_targets = train_targets[nn_indices]
    K = muygps.kernel(pairwise_tensor(train_features, nn_indices))

    precomputed_coefficients_matrix = muygps.fast_coefficients(
        K, train_nn_targets
    )

    return precomputed_coefficients_matrix, nn_indices


def make_fast_multivariate_regressor(
    mmuygps: MMuyGPS,
    nbrs_lookup: NN_Wrapper,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Convenience function for creating precomputed coefficient matrix and neighbor lookup data
    structure.

    Args:
        muygps:
            A trained MultivariateMuyGPS object.
        nbrs_lookup:
             A data structure supporting nearest neighbor queries into
            `train_features`.
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.

    Returns
    -------
    precomputed_coefficients_matrix:
        A matrix of shape `(train_count, nn_count)` whose rows list the
        precomputed coefficients for each nearest neighbors set in the
        training data.
    nn_indices:
        An array supporting nearest neighbor queries.
    """
    num_training_samples, _ = train_features.shape
    nn_indices, _ = nbrs_lookup.get_batch_nns(
        mm.arange(0, num_training_samples)
    )

    nn_indices = fast_nn_update(nn_indices)
    pairwise_diffs_fast = pairwise_tensor(train_features, nn_indices)
    train_nn_targets = train_targets[nn_indices]
    precomputed_coefficients_matrix = mmuygps.fast_coefficients(
        pairwise_diffs_fast, train_nn_targets
    )
    return precomputed_coefficients_matrix, nn_indices


def _decide_and_make_fast_regressor(
    muygps: Union[MuyGPS, MMuyGPS],
    nbrs_lookup: NN_Wrapper,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray]:
    if isinstance(muygps, MuyGPS):
        precomputed_coefficients_matrix, nn_indices = make_fast_regressor(
            muygps,
            nbrs_lookup,
            train_features,
            train_targets,
        )
    else:
        (
            precomputed_coefficients_matrix,
            nn_indices,
        ) = make_fast_multivariate_regressor(
            muygps,
            nbrs_lookup,
            train_features,
            train_targets,
        )
    return precomputed_coefficients_matrix, nn_indices


def do_fast_posterior_mean(
    test_features: mm.ndarray,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_fn: LossFn = lool_fn,
    opt_fn: OptimizeFn = Bayes_optimize,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[mm.ndarray, NN_Wrapper, mm.ndarray, mm.ndarray, Dict]:
    """
    Convenience function initializing a model and performing fast posterior mean
    inference.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Also supports workflows relying upon multivariate models. In order to create
    a multivariate model, specify the `kern` argument and pass a list of
    hyperparameter dicts to `k_kwargs`.

    Example:
        >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
        >>> from MuyGPyS.examples.fast_posterior_mean import do_fast_posterior_mean
        >>> from MuyGPyS.gp.deformation import F2, Isotropy
        >>> from MuyGPyS.gp.hyperparameter import Parameter
        >>> from MuyGPyS.gp.hyperparameter import AnalyticScale
        >>> from MuyGPyS.gp.kernels import RBF
        >>> from MuyGPyS.gp.noise import HomoscedasticNoise
        >>> from MuyGPyS.optimize import Bayes_optimize
        >>> from MuyGPyS.optimize.objective import mse_fn
        >>> train_features, train_responses = make_train()  # stand-in function
        >>> test_features, test_responses = make_test()  # stand-in function
        >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
        >>> k_kwargs = {
        ...     "kernel": RBF(
        ...         deformation=Isotropy(
        ...             metric=F2,
        ...             length_scale=Parameter(1.0, (1e-2, 1e2))
        ...         )
        ...     ),
        ...     "noise": HomoscedasticNoise(1e-5),
        ...     "scale": AnalyticScale(),
        ... }
        >>> (
        ...     muygps, nbrs_lookup, predictions, precomputed_coefficients_matrix
        ... ) = do_fast_posterior_mean(
        ...         test_features,
        ...         train_features,
        ...         train_responses,
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_fn=lool_fn,
        ...         opt_fn=Bayes_optimize,
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )

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
        loss_fn:
            The loss functor to use in hyperparameter optimization. Ignored if
            all of the parameters specified by argument `k_kwargs` are fixed.
        opt_fn:
            The optimization functor to use in hyperparameter optimization.
            Ignored if all of the parameters specified by argument `k_kwargs`
            are fixed.
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
    precomputed_coefficients_matrix:
        A matrix of shape `(train_count, nn_count)` whose rows list the
        precomputed coefficients for each nearest neighbors set in the
        training data.
    timing:
        A dictionary containing timings for the training, precomputation,
        nearest neighbor computation, and prediction.
    """
    regressor, nbrs_lookup = _decide_and_make_regressor(
        train_features,
        train_targets,
        nn_count=nn_count,
        batch_count=batch_count,
        loss_fn=loss_fn,
        opt_fn=opt_fn,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        opt_kwargs=opt_kwargs,
        verbose=verbose,
    )

    (
        posterior_mean,
        precomputed_coefficients_matrix,
        timing,
    ) = fast_posterior_mean_any(
        regressor,
        test_features,
        train_features,
        nbrs_lookup,
        train_targets,
    )
    return (
        regressor,
        nbrs_lookup,
        posterior_mean,
        precomputed_coefficients_matrix,
        timing,
    )


def fast_posterior_mean_any(
    muygps: Union[MuyGPS, MMuyGPS],
    test_features: mm.ndarray,
    train_features: mm.ndarray,
    nbrs_lookup: NN_Wrapper,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray, Dict]:
    """
    Convenience function performing fast posterior mean inference using a
    pre-trained model.

    Also supports workflows relying upon multivariate models.

    Args:
        muygps:
            A (possibly trained) MuyGPS object.
        test_features:
            A matrix of shape `(test_count, feature_count)` whose rows consist
            of observation vectors of the test data.
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        nbrs_lookup:
             A data structure supporting nearest neighbor queries into
            `train_features`.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.


    Returns
    -------
    posterior_mean:
        The predicted response associated with each test observation.
    precomputed_coefficients_matrix:
        A matrix of shape `(train_count, nn_count)` whose rows list the
        precomputed coefficients for each nearest neighbors set in the
        training data.
    timing:
        A dictionary containing timings for the training, precomputation,
        nearest neighbor computation, and prediction.

    """
    time_start = perf_counter()
    (
        precomputed_coefficients_matrix,
        nn_indices,
    ) = _decide_and_make_fast_regressor(
        muygps,
        nbrs_lookup,
        train_features,
        train_targets,
    )
    time_precomp = perf_counter()

    time_agree = perf_counter()
    nn_indices = fast_nn_update(nn_indices)

    test_neighbors, _ = nbrs_lookup.get_nns(test_features)
    time_nn = perf_counter()

    closest_neighbor = test_neighbors[:, 0]
    closest_set_new = nn_indices[closest_neighbor, :].astype(int)
    num_test_samples, _ = test_features.shape

    posterior_mean = fast_posterior_mean_from_indices(
        muygps,
        mm.arange(0, num_test_samples),
        closest_set_new,
        test_features,
        train_features,
        closest_neighbor,
        precomputed_coefficients_matrix,
    )
    time_pred = perf_counter()

    timing = {
        "precompute": time_precomp - time_start,
        "agree": time_agree - time_precomp,
        "nn": time_nn - time_agree,
        "pred": time_pred - time_nn,
    }

    return posterior_mean, precomputed_coefficients_matrix, timing
