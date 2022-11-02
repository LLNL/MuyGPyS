# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Resources and high-level API for a fast regression workflow.

:func:`~MuyGPyS.examples.regress.make_fast_regressor` is a high-level API for
creating the necessary components for fast regression.
:func:`~MuyGPyS.examples.regress.make_fast_multivariate_regressor` is a high-level
API for creating the necessary components for fast regression with multiple
outputs. 

:func:`~MuyGPyS.examples.regress.do_fast_regress` is a high-level api 
for executing a simple, generic regression workflow given data. 
It calls the maker APIs above and 
:func:`~MuyGPyS.examples.fast_regress.fast_regress_any`.
"""

import numpy as np

from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

from MuyGPyS.gp.distance import (
    make_train_tensors,
    crosswise_distances,
)
from MuyGPyS.optimize.chassis import optimize_from_tensors

from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.sigma_sq import (
    muygps_sigma_sq_optim,
    mmuygps_sigma_sq_optim,
)

from MuyGPyS.examples.regress import (
    _decide_and_make_regressor,
    _unpack,
)


def make_fast_regressor(
    muygps: MuyGPS,
    nbrs_lookup: NN_Wrapper,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Convenience function for creating precomputed coefficient matrix and neighbor lookup data
    structure.

    Args:
        muygps:
            A trained MuyGPS object.
        nbrs_lookup:
            A nearest neighbor lookup structure.
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
        np.arange(0, num_training_samples)
    )
    nn_indices = np.array(nn_indices).astype(int)

    precomputed_coefficients_matrix = muygps.build_fast_regress_coeffs(
        train_features, nn_indices, train_targets
    )
    return precomputed_coefficients_matrix, nn_indices


def make_fast_multivariate_regressor(
    muygps: MMuyGPS,
    nbrs_lookup: NN_Wrapper,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Convenience function for creating precomputed coefficient matrix and neighbor lookup data
    structure.

    Args:
        muygps:
            A trained MultivariateMuyGPS object.
        nbrs_lookup:
            A nearest neighbor lookup structure.
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
        np.arange(0, num_training_samples)
    )

    precomputed_coefficients_matrix = muygps.build_fast_regress_coeffs(
        train_features, nn_indices, train_targets
    )
    return precomputed_coefficients_matrix, nn_indices


def _decide_and_make_fast_regressor(
    muygps: Union[MuyGPS, MMuyGPS],
    nbrs_lookup: NN_Wrapper,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> np.ndarray:
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


def do_fast_regress(
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "lool",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    sigma_method: Optional[str] = "analytic",
    kern: Optional[str] = None,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        >>> from MuyGPyS.examples.fast_regress import do_fast_regress
        >>> from MuyGPyS.optimize.objective import mse_fn
        >>> train, test = _make_gaussian_data(10000, 1000, 100, 10)
        >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
        >>> k_kwargs = {
        ...         "kern": "rbf",
        ...         "metric": "F2",
        ...         "eps": {"val": 1e-5},
        ...         "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)}
        ... }
        >>> muygps, nbrs_lookup, predictions, precomputed_coefficients_matrix
        ...         = do_fast_regress(
        ...         test['input'],
        ...         train['input'],
        ...         train['output'],
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
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

    predictions, precomputed_coefficients_matrix = fast_regress_any(
        regressor,
        test_features,
        train_features,
        nbrs_lookup,
        train_targets,
    )
    return regressor, nbrs_lookup, predictions, precomputed_coefficients_matrix


def fast_regress_any(
    muygps: Union[MuyGPS, MMuyGPS],
    test_features: np.ndarray,
    train_features: np.ndarray,
    nbrs_lookup: NN_Wrapper,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Convenience function performing regression using a pre-trained model.

    Also supports workflows relying upon multivariate models.

    Args:
        muygps:
            A trained MuyGPS object.
        test_features:
            A matrix of shape `(test_count, feature_count)` whose rows consist
            of observation vectors of the test data.
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        nbrs_lookup:
            A nearest neighbor data structure NN_Wrapper object.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of response vectors of the train data.


    Returns
    -------
    predictions:
        The predicted response associated with each test observation.
    precomputed_coefficients_matrix:
        A matrix of shape `(train_count, nn_count)` whose rows list the
        precomputed coefficients for each nearest neighbors set in the
        training data.
    """

    (
        precomputed_coefficients_matrix,
        nn_indices,
    ) = _decide_and_make_fast_regressor(
        muygps,
        nbrs_lookup,
        train_features,
        train_targets,
    )
    num_training_samples, _ = train_features.shape
    _, nn_count = nn_indices.shape
    nn_indices_with_self = np.zeros((num_training_samples, nn_count + 1))
    nn_indices_with_self[:, 1 : nn_count + 1] = nn_indices
    nn_indices_with_self[:, 0] = np.arange(0, num_training_samples)
    nn_indices = nn_indices_with_self[:, :-1].astype(int)

    test_neighbors, _ = nbrs_lookup.get_nns(test_features)
    closest_neighbor = test_neighbors[:, 0]
    closest_set_new = nn_indices[closest_neighbor, :].astype(int)
    num_test_samples, _ = test_features.shape

    crosswise_dist_tens = crosswise_distances(
        test_features,
        train_features,
        np.arange(0, num_test_samples),
        closest_set_new,
    )
    Kcross_test_tens = muygps.kernel(crosswise_dist_tens)
    predictions = muygps.fast_regress_from_indices(
        Kcross_test_tens,
        closest_neighbor,
        precomputed_coefficients_matrix,
    )

    return predictions, precomputed_coefficients_matrix
