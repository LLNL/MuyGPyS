# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Convenience functions for optimizing :class:`MuyGPyS.gp.muygps.MuyGPS` 
objects

Currently wraps :class:`scipy.optimize.opt` multiparameter optimization using 
the objective function :func:`MuyGPyS.optimize.objective.loo_crossval` in order
to optimize a specified subset of the hyperparameters associated with a 
:class:'MuyGPyS.gp.muygps.MuyGPS' object.
"""


import numpy as np

from copy import deepcopy
from typing import Callable, Optional

from scipy import optimize as opt

from MuyGPyS.gp.distance import make_train_tensors
from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS.optimize.objective import get_loss_func, loo_crossval


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
    ) = make_train_tensors(
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
    x0_names, x0, bounds = muygps.get_optim_params()
    fixed_kernel_params = muygps.get_fixed_kernel_params()

    kernel_fn = _make_kernel_fn(muygps.kernel.fn, x0_names, fixed_kernel_params)
    predict_fn = _make_predict_fn(
        muygps._regress, None if x0_names[-1] == "eps" else muygps.eps()
    )

    if verbose is True:
        print(f"parameters to be optimized: {x0_names}")
        print(f"bounds: {bounds}")
        print(f"initial x0: {x0}")

    optres = opt.minimize(
        loo_crossval,
        x0,
        args=(
            loss_fn,
            kernel_fn,
            predict_fn,
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

    ret = deepcopy(muygps)

    # set final values
    for i, key in enumerate(x0_names):
        lb, ub = bounds[i]
        if optres.x[i] < lb:
            val = lb
        elif optres.x[i] > ub:
            val = ub
        else:
            val = optres.x[i]
        if key == "eps":
            ret.eps._set_val(val)
        else:
            ret.kernel.hyperparameters[key]._set_val(val)

    return ret


def _make_kernel_fn(fn, x0_names, fixed_params):
    if x0_names[-1] == "eps":

        def fn_caller(dists, x0):
            return fn(
                dists, **dict(zip(x0_names[:-1], x0[:-1])), **fixed_params
            )

    else:

        def fn_caller(dists, x0):
            return fn(dists, **dict(zip(x0_names, x0)), **fixed_params)

    return fn_caller


def _make_predict_fn(fn, eps: Optional[float] = None) -> Callable:
    if eps is None:
        # Here we take eps (x0) as an argument
        def fn_caller(K, Kcross, batch_nn_targets, x0):
            return fn(K, Kcross, batch_nn_targets, x0, "unlearned")

    else:
        # In this case, ignore x0 because it is fixed
        def fn_caller(K, Kcross, batch_nn_targets, x0):
            return fn(K, Kcross, batch_nn_targets, eps, "unlearned")

    return fn_caller