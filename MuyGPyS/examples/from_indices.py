# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenience wrapper for GP prediction from indices.
"""

import numpy as np

from typing import Tuple, Union

from MuyGPyS.gp.tensors import (
    crosswise_tensor,
    make_predict_tensors,
    make_train_tensors,
)
from MuyGPyS.gp import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.optimize import Bayes_optimize, OptimizeFn
from MuyGPyS.optimize.loss import LossFn, lool_fn


def tensors_from_indices(
    muygps: Union[MuyGPS, MMuyGPS],
    indices: np.ndarray,
    nn_indices: np.ndarray,
    test: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    crosswise_tensor, pairwise_tensor, batch_nn_targets = make_predict_tensors(
        indices, nn_indices, test, train, targets
    )
    if isinstance(muygps, MuyGPS):
        pairwise_tensor = muygps.kernel(pairwise_tensor)
        crosswise_tensor = muygps.kernel(crosswise_tensor)
    return pairwise_tensor, crosswise_tensor, batch_nn_targets


def posterior_mean_from_indices(
    muygps: Union[MuyGPS, MMuyGPS],
    indices: np.ndarray,
    nn_indices: np.ndarray,
    test: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    **kwargs,
) -> np.ndarray:
    pairwise_tensor, crosswise_tensor, batch_nn_targets = tensors_from_indices(
        muygps, indices, nn_indices, test, train, targets
    )
    return muygps.posterior_mean(
        pairwise_tensor, crosswise_tensor, batch_nn_targets
    )


def posterior_variance_from_indices(
    muygps: Union[MuyGPS, MMuyGPS],
    indices: np.ndarray,
    nn_indices: np.ndarray,
    test: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    **kwargs,
) -> np.ndarray:
    pairwise_tensor, crosswise_tensor, batch_nn_targets = tensors_from_indices(
        muygps, indices, nn_indices, test, train, targets
    )
    return muygps.posterior_variance(
        pairwise_tensor, crosswise_tensor, **kwargs
    )


def regress_from_indices(
    muygps: Union[MuyGPS, MMuyGPS],
    indices: np.ndarray,
    nn_indices: np.ndarray,
    test: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    pairwise_tensor, crosswise_tensor, batch_nn_targets = tensors_from_indices(
        muygps, indices, nn_indices, test, train, targets
    )
    return muygps.posterior_mean(
        pairwise_tensor, crosswise_tensor, batch_nn_targets
    ), muygps.posterior_variance(pairwise_tensor, crosswise_tensor, **kwargs)


def fast_posterior_mean_from_indices(
    muygps: Union[MuyGPS, MMuyGPS],
    indices: np.ndarray,
    nn_indices: np.ndarray,
    test_features: np.ndarray,
    train_features: np.ndarray,
    closest_index: np.ndarray,
    coeffs_tensor: np.ndarray,
) -> np.ndarray:
    crosswise_diffs = crosswise_tensor(
        test_features,
        train_features,
        indices,
        nn_indices,
    )
    if isinstance(muygps, MuyGPS):
        Kcross = muygps.kernel(crosswise_diffs)
        return muygps.fast_posterior_mean(
            Kcross,
            coeffs_tensor[closest_index, :, :],
        )
    else:
        return muygps.fast_posterior_mean(
            crosswise_diffs,
            coeffs_tensor[closest_index, :, :],
        )


def optimize_from_indices(
    muygps: MuyGPS,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    loss_fn: LossFn = lool_fn,
    opt_fn: OptimizeFn = Bayes_optimize,
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    """
    Find an optimal model directly from the data.

    Use this method if you do not need to retain the difference and kernel
    tensors used for optimization.

    See the following example, where we have already created a `batch_indices`
    vector and a `batch_nn_indices` matrix using
    :class:`MuyGPyS.neighbors.NN_Wrapper`, and initialized a
    :class:`~MuyGPyS.gp.muygps.MuyGPS` model `muygps`.

    Example:
        >>> from MuyGPyS.optimize.chassis import optimize_from_indices
        >>> muygps = optimize_from_indices(
        ...         muygps,
        ...         batch_indices,
        ...         batch_nn_indices,
        ...         train_features,
        ...         train_features,
        ...         train_responses,
        ...         loss_fn=lool_fn,
        ...         opt_fn=L_BFGS_B_optimize,
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
        loss_fn:
            Indicates the loss functor to be used.
        opt_fn:
            The optimization functor to use in hyperparameter optimization.
        verbose:
            If True, print debug messages.
        kwargs:
            Additional keyword arguments to be passed to the wrapper optimizer.

    Returns:
        A new MuyGPs model whose specified hyperparameters have been optimized.
    """
    (
        crosswise_diffs,
        pairwise_diffs,
        batch_targets,
        batch_nn_targets,
    ) = make_train_tensors(
        batch_indices,
        batch_nn_indices,
        train_features,
        train_targets,
    )
    return opt_fn(
        muygps,
        batch_targets,
        batch_nn_targets,
        crosswise_diffs,
        pairwise_diffs,
        loss_fn=loss_fn,
        verbose=verbose,
        **kwargs,
    )
