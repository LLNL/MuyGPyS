# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenience functions for optimizing :class:`~MuyGPyS.gp.muygps.MuyGPS` objects
"""


from typing import Callable, Dict, Optional

import MuyGPyS._src.math as mm
from MuyGPyS._src.optimize.chassis import (
    _scipy_optimize,
    _bayes_opt_optimize,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.optimize.objective import make_loo_crossval_fn
from MuyGPyS.optimize.loss import lool_fn, LossFn


class OptimizeFn:
    """
    Outer-loop optimization functor class.

    MuyGPyS-compatible optimization functions are objects of this class.
    Creating a new outer-loop optimization function is as simple as
    initializing a new `OptimizeFn` object with conforming `optimize_fn` and
    `make_obj_fn` functions.

    Args:
        optimize_fn:
            A Callable with the signature
            `(muygps, obj_fn, verbose=verbose, **kwargs) -> MuyGPS`.
        make_obj_fn:
            A Callable taking the following objects, in order:
            `loss_fn`, `kernel_fn`, `mean_fn`, `var_fn`, `scale_fn`,
            `pairwise_diffs`, `crosswise_diffs` `batch_nn_targets`,
            `batch_targets` `batch_features`, `loss_kwargs`.
    """

    def __init__(self, optimize_fn: Callable, make_obj_fn: Callable):
        self._fn = optimize_fn
        self._make_obj_fn = make_obj_fn

    def __call__(
        self,
        muygps: MuyGPS,
        batch_targets: mm.ndarray,
        batch_nn_targets: mm.ndarray,
        crosswise_diffs: mm.ndarray,
        pairwise_diffs: mm.ndarray,
        batch_features: Optional[mm.ndarray] = None,
        loss_fn: LossFn = lool_fn,
        loss_kwargs: Dict = dict(),
        target_mask: Optional[mm.ndarray] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Find the optimal model using existing difference matrices.

        Args:
            muygps:
                The model to be optimized.
            batch_targets:
                Matrix of floats of shape `(batch_count,) [+ (response_count,)]`
                listing the expected (possibly multivariate) responses for each
                batch element.
            batch_nn_targets:
                Tensor of floats of shape
                `(batch_count, nn_count) [+ (response_count,)]` containing the
                expected (possibly multivariate) response for each nearest
                neighbor of each batch element.
            crosswise_diffs:
                A tensor of shape
                `(batch_count, nn_count) [+ (feature_count,)]` containing the
                crosswise distances or feature-dimension-wise differences
                (extra `feature_count` dimension) between the batch elements
                and each of their nearest neighbors.
            pairwise_diffs:
                A tensor of shape
                `(batch_count, nn_count, nn_count) [+ (feature_count,)]`
                containing the pairwise distances or feature-dimension-wise
                differences (extra `feature_count` dimension) between all pairs
                of nearest neighbors for each batch element.
            loss_fn:
                The loss functor used to evaluate model performance.
            loss_kwargs:
                A dictionary of additional keyword arguments to apply to the
                :class:`~MuyGPyS.optimize.loss.LossFn`. Loss function specific.
            target_mask:
                An array of indices, listing the output dimensions of the
                prediction to be used for optimization.
            verbose:
                If True, print debug messages.
            kwargs:
                Additional keyword arguments to be passed to the wrapper
                optimizer.

        Returns:
            A new MuyGPs model whose specified hyperparameters have been
            optimized.
        """
        obj_fn = self.make_obj_fn(
            muygps,
            batch_targets,
            batch_nn_targets,
            crosswise_diffs,
            pairwise_diffs,
            batch_features=batch_features,
            target_mask=target_mask,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        )
        return self._fn(muygps, obj_fn, verbose=verbose, **kwargs)

    def make_obj_fn(
        self,
        muygps: MuyGPS,
        batch_targets: mm.ndarray,
        batch_nn_targets: mm.ndarray,
        crosswise_diffs: mm.ndarray,
        pairwise_diffs: mm.ndarray,
        batch_features: Optional[mm.ndarray] = None,
        target_mask: Optional[mm.ndarray] = None,
        loss_fn: LossFn = lool_fn,
        loss_kwargs: Dict = dict(),
        **kwargs,
    ) -> Callable:
        """
        Returns the objective function specified by the training batch and
        model choices.

        Args:
            muygps:
                The model to be optimized.
            batch_targets:
                Matrix of floats of shape `(batch_count,) [+ (response_count,)]`
                listing the expected (possibly multivariate) responses for each
                batch element.
            batch_nn_targets:
                Tensor of floats of shape
                `(batch_count, nn_count) [+ (response_count,)]` containing the
                expected (possibly multivariate) response for each nearest
                neighbor of each batch element.
            crosswise_diffs:
                A tensor of shape
                `(batch_count, nn_count) [+ (feature_count,)]` containing the
                crosswise distances or feature-dimension-wise differences
                (extra `feature_count` dimension) between the batch elements
                and each of their nearest neighbors.
            pairwise_diffs:
                A tensor of shape
                `(batch_count, nn_count, nn_count) [+ (feature_count,)]`
                containing the pairwise distances or feature-dimension-wise
                differences (extra `feature_count` dimension) between all pairs
                of nearest neighbors for each batch element.
            loss_fn:
                The loss functor used to evaluate model performance.
            target_mask:
                An array of indices, listing the output dimensions of the
                prediction to be used for optimization.
            loss_kwargs:
                A dictionary of additional keyword arguments to apply to the
                :class:`~MuyGPyS.optimize.loss.LossFn`. Loss function specific.
            kwargs:
                Additional keyword arguments to be passed to the wrapper
                optimizer.

        Returns:
            A Callable function that evaluates the objective function for a
            given value of the free parameters.
        """
        kernel_fn = muygps.kernel.get_opt_fn()
        mean_fn = muygps.get_opt_mean_fn()
        var_fn = muygps.get_opt_var_fn()
        scale_fn = muygps.scale.get_opt_fn(muygps)

        return self._make_obj_fn(
            loss_fn,
            kernel_fn,
            mean_fn,
            var_fn,
            scale_fn,
            pairwise_diffs,
            crosswise_diffs,
            batch_nn_targets,
            batch_targets,
            batch_features=batch_features,
            target_mask=target_mask,
            loss_kwargs=loss_kwargs,
        )


Bayes_optimize = OptimizeFn(_bayes_opt_optimize, make_loo_crossval_fn)
"""
Optimize a :class:`~MuyGPyS.gp.muygps.MuyGPS` model using Bayesian optimization.

See the following example, where we have already created a
`batch_indices` vector and a `batch_nn_indices` matrix using
:class:`MuyGPyS.neighbors.NN_Wrapper`, a `crosswise_diffs`
matrix using :func:`MuyGPyS.gp.tensors.crosswise_tensor` and
`pairwise_diffs` using :func:`MuyGPyS.gp.tensors.pairwise_tensor`, and
initialized a :class:`~MuyGPyS.gp.muygps.MuyGPS` model `muygps`.

Example:
    >>> from MuyGPyS.optimize import Bayes_optimize
    >>> muygps = Bayes_optimize(
    ...         muygps,
    ...         batch_targets,
    ...         batch_nn_targets,
    ...         crosswise_diffs,
    ...         pairwise_diffs,
    ...         train_responses,
    ...         loss_fn=lool_fn,
    ...         verbose=True,
    ... )
    parameters to be optimized: ['nu']
    bounds: [[0.1 5. ]]
    initial x0: [0.92898658]
    |   iter    |  target   |    nu     |
    -------------------------------------
    | 1         | 1.826e+03 | 0.929     |
    | 2         | 2.359e+03 | 2.143     |
    | 3         | 1.953e+03 | 3.63      |
    | 4         | 614.4     | 0.1006    |
    | 5         | 2.309e+03 | 1.581     |
    | 6         | 1.707e+03 | 0.8191    |
    | 7         | 1.48e+03  | 5.0       |
    | 8         | 2.202e+03 | 2.83      |
    | 9         | 2.373e+03 | 1.883     |
    | 10        | 2.373e+03 | 1.996     |
    | 11        | 2.375e+03 | 1.938     |
    | 12        | 2.375e+03 | 1.938     |
    | 13        | 2.375e+03 | 1.938     |
    | 14        | 2.375e+03 | 1.938     |
    | 15        | 2.375e+03 | 1.938     |
    | 16        | 2.375e+03 | 1.938     |
    | 17        | 2.375e+03 | 1.938     |
    | 18        | 2.375e+03 | 1.945     |
    | 19        | 2.375e+03 | 1.927     |
    | 20        | 2.375e+03 | 1.95      |
    | 21        | 2.375e+03 | 1.926     |
    =====================================

Args:
    muygps:
        The model to be optimized.
    batch_targets:
        Matrix of floats of shape `(batch_count,) [+ (response_count,)]` listing
        the expected (possibly multivariate) responses for each batch element.
    batch_nn_targets:
        Tensor of floats of shape
        `(batch_count, nn_count) [+ (response_count,)]` containing the expected
        (possibly multivariate) response for each nearest neighbor of each batch
        element.
    crosswise_diffs:
        A tensor of shape
        `(batch_count, nn_count) [+ (feature_count,)]` containing the crosswise
        distances or feature-dimension-wise differences (extra `feature_count`
        dimension) between the batch elements and each of their nearest
        neighbors.
    pairwise_diffs:
        A tensor of shape
        `(batch_count, nn_count, nn_count) [+ (feature_count,)]` containing the
        pairwise distances or feature-dimension-wise differences (extra
        `feature_count` dimension) between all pairs of nearest neighbors for
        each batch element.
    loss_fn:
        The loss functor used to evaluate model performance.
    loss_kwargs:
        A dictionary of additional keyword arguments to apply to the
        :class:`~MuyGPyS.optimize.loss.LossFn`. Loss function specific.
    verbose:
        If True, print debug messages.
    kwargs:
        Additional keyword arguments to be passed to the wrapper
        optimizer.

Returns:
    A new MuyGPs model whose specified hyperparameters have been
    optimized.
"""

L_BFGS_B_optimize = OptimizeFn(_scipy_optimize, make_loo_crossval_fn)
"""
Optimize a :class:`~MuyGPyS.gp.muygps.MuyGPS` model using the L-BFGS-B
algorithm.

See the following example, where we have already created a
`batch_indices` vector and a `batch_nn_indices` matrix using
:class:`MuyGPyS.neighbors.NN_Wrapper`, a `crosswise_diffs`
matrix using :func:`MuyGPyS.gp.tensors.crosswise_tensor` and
`pairwise_diffs` using :func:`MuyGPyS.gp.tensors.pairwise_tensor`, and
initialized a :class:`~MuyGPyS.gp.muygps.MuyGPS` model `muygps`.

Example:
    >>> from MuyGPyS.optimize import L_BFGS_B_optimize
    >>> muygps = L_BFGS_B_optimize(
    ...         muygps,
    ...         batch_targets,
    ...         batch_nn_targets,
    ...         crosswise_diffs,
    ...         pairwise_diffs,
    ...         train_responses,
    ...         loss_fn=lool_fn,
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
        Matrix of floats of shape `(batch_count,) [+ (response_count,)]` listing
        the expected (possibly multivariate) responses for each batch element.
    batch_nn_targets:
        Tensor of floats of shape
        `(batch_count, nn_count) [+ (response_count,)]` containing the expected
        (possibly multivariate) response for each nearest neighbor of each batch
        element.
    crosswise_diffs:
        A tensor of shape
        `(batch_count, nn_count) [+ (feature_count,)]` containing the crosswise
        distances or feature-dimension-wise differences (extra `feature_count`
        dimension) between the batch elements and each of their nearest
        neighbors.
    pairwise_diffs:
        A tensor of shape
        `(batch_count, nn_count, nn_count) [+ (feature_count,)]` containing the
        pairwise distances or feature-dimension-wise differences (extra
        `feature_count` dimension) between all pairs of nearest neighbors for
        each batch element.
    loss_fn:
        The loss functor used to evaluate model performance.
    loss_kwargs:
        A dictionary of additional keyword arguments to apply to the
        :class:`~MuyGPyS.optimize.loss.LossFn`. Loss function specific.
    verbose:
        If True, print debug messages.
    kwargs:
        Additional keyword arguments to be passed to the wrapper
        optimizer.

Returns:
    A new MuyGPs model whose specified hyperparameters have been
    optimized.
"""
