# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenience functions for optimizing :class:`MuyGPyS.gp.muygps.MuyGPS` objects

Currently wraps :class:`scipy.optimize.opt` multiparameter optimization using
the objective function :func:`MuyGPyS.optimize.objective.loo_crossval` in order
to optimize a specified subset of the hyperparameters associated with a
:class:'MuyGPyS.gp.muygps.MuyGPS' object.
"""


import numpy as np

from MuyGPyS import config

if config.jax_enabled() is False:
    from MuyGPyS._src.gp.numpy_distance import _make_train_tensors
    from MuyGPyS._src.optimize.numpy_chassis import _scipy_optimize_from_tensors
else:
    from MuyGPyS._src.gp.jax_distance import _make_train_tensors
    from MuyGPyS._src.optimize.jax_chassis import _scipy_optimize_from_tensors

from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS.optimize.objective import get_loss_func, make_loo_crossval_fn


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
    Find an optimal model using scipy directly from the data.

    Use this method if you do not need to retain the distance matrices used for
    optimization.

    See the following example, where we have already created a `batch_indices`
    vector and a `batch_nn_indices` matrix using
    :class:`MuyGPyS.neighbors.NN_Wrapper`, and initialized a
    :class:`MuyGPyS.gp.muygps.MuyGPS` model `muygps`.

    Example:
        >>> from MuyGPyS.optimize.chassis import scipy_optimize_from_indices
        >>> muygps = scipy_optimize_from_indices(
        ...         muygps,
        ...         batch_indices,
        ...         batch_nn_indices,
        ...         train_features,
        ...         train_features,
        ...         train_responses,
        ...         loss_method='mse',
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
        verbose : bool
            If True, print debug messages.

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
    return scipy_optimize_from_tensors(
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
        >>> from MuyGPyS.optimize.chassis import scipy_optimize_from_tensors
        >>> muygps = scipy_optimize_from_tensors(
        ...         muygps,
        ...         batch_indices,
        ...         batch_nn_indices,
        ...         crosswise_dists,
        ...         pairwise_dists,
        ...         train_responses,
        ...         loss_method='mse',
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
        verbose:
            If True, print debug messages.

    Returns:
        A new MuyGPs model whose specified hyperparameters have been optimized.
    """
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

    return _scipy_optimize_from_tensors(muygps, obj_fn, verbose=verbose)
