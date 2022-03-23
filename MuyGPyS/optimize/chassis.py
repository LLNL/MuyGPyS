# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenience functions for optimizing :class:`MuyGPyS.gp.muygps.MuyGPS` objects

The functions
:func:`~MuyGPyS.optimize.chassis.optimize_from_indices` and
:func:`~MuyGPyS.optimize.chassis.optimize_from_tensors` wrap different
optimization packages to provide a simple interface to optimize the
hyperparameters of :class:`~MuyGPyS.gp.muygps.MuyGPS` objects.

Currently, `opt_method="scipy"` wraps :class:`scipy.optimize.opt`
multiparameter optimization using L-BFGS-B algorithm using the objective
function :func:`MuyGPyS.optimize.objective.loo_crossval`.

Currently, `opt_method="bayesian"` (also accepts `"bayes"` and `"bayes_opt"`)
wraps :class:`bayes_opt.BayesianOptimization`. Unlike the `scipy` version,
`BayesianOptimization` can be meaningfully modified by several kwargs.
`MuyGPyS` assigns reasonable defaults if no settings are passed by the user.
See the `BayesianOptimization <https://github.com/fmfn/BayesianOptimization>`_
documentation for details.
"""


import numpy as np
import warnings

from MuyGPyS import config

if config.muygpys_jax_enabled is False:  # type: ignore
    from MuyGPyS._src.gp.numpy_distance import _make_train_tensors
    from MuyGPyS._src.optimize.numpy_chassis import (
        _scipy_optimize,
        _bayes_opt_optimize,
    )

else:
    from MuyGPyS._src.gp.jax_distance import _make_train_tensors
    from MuyGPyS._src.optimize.jax_chassis import (
        _scipy_optimize,
        _bayes_opt_optimize,
    )

from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS.optimize.objective import (
    get_loss_func,
    make_loo_crossval_fn,
    make_loo_crossval_kwargs_fn,
)


def optimize_from_indices(
    muygps: MuyGPS,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    loss_method: str = "mse",
    opt_method: str = "scipy",
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    """
    Find an optimal model directly from the data.

    Use this method if you do not need to retain the distance matrices used for
    optimization.

    See the following example, where we have already created a `batch_indices`
    vector and a `batch_nn_indices` matrix using
    :class:`MuyGPyS.neighbors.NN_Wrapper`, and initialized a
    :class:`MuyGPyS.gp.muygps.MuyGPS` model `muygps`.

    Example:
        >>> from MuyGPyS.optimize.chassis import optimize_from_indices
        >>> muygps = optimize_from_indices(
        ...         muygps,
        ...         batch_indices,
        ...         batch_nn_indices,
        ...         train_features,
        ...         train_features,
        ...         train_responses,
        ...         loss_method='mse',
        ...         opt_method='scipy',
        ...         verbose=True,
        ... )
        parameters to be optimized: ['nu']
        bounds: [[0.1 1. ]]
        sampled x0: [0.8858425]
        optimizer results:
              fun: 0.4797763813693626
         hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
              jac: array([-3.06976666e-06])
          message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
             nfev: 16
              nit: 5
             njev: 8
           status: 0
          success: True
                x: array([0.39963594])

    Args:
        muygps:
            The model to be optimized.
        batch_indices:
            A vector of integers of shape `(batch_count,)` identifying the
            training batch of observations to be approximated.
        batch_nn_indices:
            A matrix of integers of shape `(batch_count, nn_count)` listing the
            nearest neighbor indices for all observations in the batch.
        train_features:
            The full floating point training data matrix of shape
            `(train_count, feature_count)`.
        train_targets:
            A matrix of shape `(train_count, feature_count)` whose rows are
            vector-valued responses for each training element.
        loss_method:
            Indicates the loss function to be used.
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` (alternately `"bayes"` or `"bayes_opt"`) and
            `"scipy"`.
        verbose:
            If True, print debug messages.
        kwargs:
            Additional keyword arguments to be passed to the wrapper optimizer.

    Returns:
        A new MuyGPs model whose specified hyperparameters have been optimized.
    """
    (
        crosswise_dists,
        pairwise_dists,
        batch_targets,
        batch_nn_targets,
    ) = _make_train_tensors(
        muygps.kernel.metric,
        batch_indices,
        batch_nn_indices,
        train_features,
        train_targets,
    )
    return optimize_from_tensors(
        muygps,
        batch_targets,
        batch_nn_targets,
        crosswise_dists,
        pairwise_dists,
        loss_method=loss_method,
        opt_method=opt_method,
        verbose=verbose,
        **kwargs,
    )


def optimize_from_tensors(
    muygps: MuyGPS,
    batch_targets: np.ndarray,
    batch_nn_targets: np.ndarray,
    crosswise_dists: np.ndarray,
    pairwise_dists: np.ndarray,
    loss_method: str = "mse",
    opt_method: str = "scipy",
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    """
    Find the optimal model using existing distance matrices.

    Use this method if you need to retain the distance matrices used for later
    use.

    See the following example, where we have already created a `batch_indices`
    vector and a `batch_nn_indices` matrix using
    :class:`MuyGPyS.neighbors.NN_Wrapper`, a `crosswise_dists`
    matrix using :func:`MuyGPyS.gp.distance.crosswise_distances` and
    `pairwise_dists` using :func:`MuyGPyS.gp.distance.pairwise_distances`, and
    initialized a :class:`MuyGPyS.gp.muygps.MuyGPS` model `muygps`.

    Example:
        >>> from MuyGPyS.optimize.chassis import optimize_from_tensors
        >>> muygps = optimize_from_tensors(
        ...         muygps,
        ...         batch_indices,
        ...         batch_nn_indices,
        ...         crosswise_dists,
        ...         pairwise_dists,
        ...         train_responses,
        ...         loss_method='mse',
        ...         opt_method='scipy',
        ...         verbose=True,
        ... )
        parameters to be optimized: ['nu']
        bounds: [[0.1 1. ]]
        sampled x0: [0.8858425]
        optimizer results:
              fun: 0.4797763813693626
         hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
              jac: array([-3.06976666e-06])
          message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
             nfev: 16
              nit: 5
             njev: 8
           status: 0
          success: True
                x: array([0.39963594])

    Args:
        muygps:
            The model to be optimized.
        batch_targets:
            Matrix of floats of shape `(batch_count, response_count)` whose rows
            give the expected response for each batch element.
        batch_nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, response_count)`
            containing the expected response for each nearest neighbor of each
            batch element.
        crosswise_dists:
            Distance matrix of floats of shape `(batch_count, nn_count)` whose
            rows give the distances between each batch element and its nearest
            neighbors.
        pairwise_dists:
            Distance tensor of floats of shape
            `(batch_count, nn_count, nn_count)` whose second two dimensions give
            the pairwise distances between the nearest neighbors of each batch
            element.
        loss_method:
            Indicates the loss function to be used.
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` (alternately `"bayes"` or `"bayes_opt"`) and
            `"scipy"`.
        verbose:
            If True, print debug messages.
        kwargs:
            Additional keyword arguments to be passed to the wrapper optimizer.

    Returns:
        A new MuyGPs model whose specified hyperparameters have been optimized.
    """
    loss_method = loss_method.lower()
    opt_method = opt_method.lower()

    if opt_method == "scipy":
        return _scipy_optimize_from_tensors(
            muygps,
            batch_targets,
            batch_nn_targets,
            crosswise_dists,
            pairwise_dists,
            loss_method=loss_method,
            verbose=verbose,
            **kwargs,
        )
    if opt_method in ["bayesian", "bayes", "bayes-opt"]:
        return _bayes_opt_optimize_from_tensors(
            muygps,
            batch_targets,
            batch_nn_targets,
            crosswise_dists,
            pairwise_dists,
            loss_method=loss_method,
            verbose=verbose,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported optimization method: {opt_method}")


def scipy_optimize_from_indices(
    muygps: MuyGPS,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    loss_method: str = "mse",
    verbose: bool = False,
) -> MuyGPS:
    """
    Find the optimal model with scipy directly from the data.

    Deprecated and will be removed in v0.6.0. Use
    `func:~MuyGPyS.optimize.chassis.optimize_from_indices()` with
    `opt_method="scipy"` instead.
    """
    warnings.warn(
        "scipy_optimize_from_indices() is deprecated, and will be removed in "
        "v0.6.0. "
        'Use optimize_from_indices() with opt_method="scipy" instead.',
        DeprecationWarning,
    )
    (
        crosswise_dists,
        pairwise_dists,
        batch_targets,
        batch_nn_targets,
    ) = _make_train_tensors(
        muygps.kernel.metric,
        batch_indices,
        batch_nn_indices,
        train_features,
        train_targets,
    )
    return _scipy_optimize_from_tensors(
        muygps,
        batch_targets,
        batch_nn_targets,
        crosswise_dists,
        pairwise_dists,
        loss_method=loss_method,
        verbose=verbose,
    )


def scipy_optimize_from_tensors(
    muygps: MuyGPS,
    batch_targets: np.ndarray,
    batch_nn_targets: np.ndarray,
    crosswise_dists: np.ndarray,
    pairwise_dists: np.ndarray,
    loss_method: str = "mse",
    verbose: bool = False,
) -> MuyGPS:
    """
    Find the optimal model with scipy using existing distance matrices.

    Deprecated and will be removed in v0.6.0. Use
    `func:~MuyGPyS.optimize.chassis.optimize_from_tensors()` with
    `opt_method="scipy"` instead.
    """
    warnings.warn(
        "scipy_optimize_from_tensors() is deprecated, and will be removed in "
        "v0.6.0. "
        'Use optimize_from_tensors() with opt_method="scipy" instead.',
        DeprecationWarning,
    )
    return _scipy_optimize_from_tensors(
        muygps,
        batch_targets,
        batch_nn_targets,
        crosswise_dists,
        pairwise_dists,
        loss_method=loss_method,
        verbose=verbose,
    )


def _scipy_optimize_from_tensors(
    muygps: MuyGPS,
    batch_targets: np.ndarray,
    batch_nn_targets: np.ndarray,
    crosswise_dists: np.ndarray,
    pairwise_dists: np.ndarray,
    loss_method: str = "mse",
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    loss_fn = get_loss_func(loss_method)

    kernel_fn = muygps.kernel.get_opt_fn()
    predict_fn = muygps.get_opt_fn()

    obj_fn = make_loo_crossval_fn(
        loss_fn,
        kernel_fn,
        predict_fn,
        pairwise_dists,
        crosswise_dists,
        batch_nn_targets,
        batch_targets,
    )

    return _scipy_optimize(muygps, obj_fn, verbose=verbose, **kwargs)


def _bayes_opt_optimize_from_tensors(
    muygps: MuyGPS,
    batch_targets: np.ndarray,
    batch_nn_targets: np.ndarray,
    crosswise_dists: np.ndarray,
    pairwise_dists: np.ndarray,
    loss_method: str = "mse",
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    loss_fn = get_loss_func(loss_method)

    kernel_fn = muygps.kernel.get_kwargs_opt_fn()
    predict_fn = muygps.get_kwargs_opt_fn()

    obj_fn = make_loo_crossval_kwargs_fn(
        loss_fn,
        kernel_fn,
        predict_fn,
        pairwise_dists,
        crosswise_dists,
        batch_nn_targets,
        batch_targets,
    )

    return _bayes_opt_optimize(muygps, obj_fn, verbose=verbose, **kwargs)
