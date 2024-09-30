# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Objective Handling

MuyGPyS includes predefined objective functions and convenience functions for
indicating them to optimization.
"""

from typing import Callable, Dict, Optional

import MuyGPyS._src.math as mm

from MuyGPyS.optimize.loss import LossFn


def make_loo_crossval_fn(
    loss_fn: LossFn,
    kernel_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    scale_fn: Callable,
    pairwise_diffs: mm.ndarray,
    crosswise_diffs: mm.ndarray,
    batch_nn_targets: mm.ndarray,
    batch_targets: mm.ndarray,
    batch_features: Optional[mm.ndarray] = None,
    target_mask: Optional[mm.ndarray] = None,
    loss_kwargs: Dict = dict(),
) -> Callable:
    """
    Prepare a leave-one-out cross validation function as a function purely of
    the hyperparameters to be optimized.

    This function is designed for use with
    :class:`MuyGPyS.optimize.chassis.OptimizeFn`.

    Args:
        loss_fn:
            The loss functor used to evaluate model performance.
        kernel_fn:
            A function that realizes kernel tensors given a list of the free
            parameters.
        mean_fn:
            A function that realizes MuyGPs posterior mean prediction given a
            noise model.
        var_fn:
            A function that realizes MuyGPs posterior variance prediction given
            a noise model.
        scale_fn:
            A function that realizes variance scale parameter optimization given
            a noise model.
        pairwise_diffs:
            A tensor of shape
            `(batch_count, nn_count, nn_count) [+ (feature_count,)]` containing
            the pairwise distances or feature-dimension-wise differences (extra
            `feature_count` dimension) between all pairs of nearest neighbors
            for each batch element.
        crosswise_diffs:
            A tensor of shape
            `(batch_count, nn_count) [+ (feature_count,)]` containing the
            crosswise distances or feature-dimension-wise differences (extra
            `feature_count` dimension) between the batch elements and each of
            their nearest neighbors.
        batch_nn_targets:
            Tensor of floats of shape
            `(batch_count, nn_count) [+ (response_count,)]` containing the
            expected (possibly multivariate) response for each nearest neighbor
            of each batch element.
        batch_targets:
            Matrix of floats of shape `(batch_count,) [+ (response_count,)`
            listing the expected (possibly multivariate) responses for each
            batch element.
        batch_features:
            Optional matrix of floats of shape `(batch_count, feature_count)`
            whose rows give the features for each batch element.
        target_mask:
            An array of indices, listing the output dimensions of the prediction
            to be used for optimization.
        loss_kwargs:
            A dict listing any additional kwargs to pass to the loss function.

    Returns:
        A Callable `objective_fn`.
    """
    kernels_fn = make_kernels_fn(kernel_fn, pairwise_diffs, crosswise_diffs)
    # This is ad-hoc, and might need to be revisited.
    predict_and_loss_fn = loss_fn.make_predict_and_loss_fn(
        mean_fn,
        var_fn,
        scale_fn,
        batch_nn_targets,
        batch_targets,
        target_mask=target_mask,
        **loss_kwargs,
    )

    def obj_fn(*args, **kwargs):
        Kin, Kcross = kernels_fn(*args, batch_features=batch_features, **kwargs)
        return predict_and_loss_fn(Kin, Kcross, *args, **kwargs)

    return obj_fn


def make_kernels_fn(
    kernel_fn: Callable,
    pairwise_diffs: mm.ndarray,
    crosswise_diffs: mm.ndarray,
) -> Callable:
    def kernels_fn(*args, **kwargs):
        Kin = kernel_fn(pairwise_diffs, *args, **kwargs)
        Kcross = kernel_fn(crosswise_diffs, *args, **kwargs)
        return Kin, Kcross

    return kernels_fn
