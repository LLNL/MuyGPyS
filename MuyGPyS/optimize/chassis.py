# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenience functions for optimizing :class:`~MuyGPyS.gp.muygps.MuyGPS` objects

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


from typing import Callable, Dict, Optional

import MuyGPyS._src.math as mm
from MuyGPyS._src.optimize.chassis import (
    _scipy_optimize,
    _bayes_opt_optimize,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.optimize.utils import _switch_on_opt_method
from MuyGPyS.optimize.objective import make_obj_fn
from MuyGPyS.optimize.loss import lool_fn
from MuyGPyS.optimize.sigma_sq import make_sigma_sq_optim


def optimize_from_tensors(
    muygps: MuyGPS,
    batch_targets: mm.ndarray,
    batch_nn_targets: mm.ndarray,
    crosswise_diffs: mm.ndarray,
    pairwise_diffs: mm.ndarray,
    batch_features: Optional[mm.ndarray] = None,
    loss_fn: Callable = lool_fn,
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    sigma_method: Optional[str] = "analytic",
    loss_kwargs: Dict = dict(),
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    """
    Find the optimal model using existing difference matrices.

    See the following example, where we have already created a `batch_indices`
    vector and a `batch_nn_indices` matrix using
    :class:`MuyGPyS.neighbors.NN_Wrapper`, a `crosswise_diffs`
    matrix using :func:`MuyGPyS.gp.tensors.crosswise_tensor` and
    `pairwise_diffs` using :func:`MuyGPyS.gp.tensors.pairwise_tensor`, and
    initialized a :class:`~MuyGPyS.gp.muygps.MuyGPS` model `muygps`.

    Example:
        >>> from MuyGPyS.optimize.chassis import optimize_from_tensors
        >>> muygps = optimize_from_tensors(
        ...         muygps,
        ...         batch_indices,
        ...         batch_nn_indices,
        ...         crosswise_diffs,
        ...         pairwise_diffs,
        ...         train_responses,
        ...         loss_method='mse',
        ...         obj_method='loo_crossval',
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
        crosswise_diffs:
            A tensor of shape `(batch_count, nn_count, feature_count)` whose
            last two dimensions list the difference between each feature of each
            batch element element and its nearest neighbors.
        pairwise_diffs:
            A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
            containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
            nearest neighbor difference tensors corresponding to each of the
            batch elements.
        loss_fn:
            The loss functor used to evaluate model performance.
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` (alternately `"bayes"` or `"bayes_opt"`) and
            `"scipy"`.
        sigma_method:
            The optimization method to be employed to learn the `sigma_sq`
            hyperparameter.
        loss_kwargs:
            A dictionary of additional keyword arguments to apply to the loss
            function. Loss function specific.
        verbose:
            If True, print debug messages.
        kwargs:
            Additional keyword arguments to be passed to the wrapper optimizer.

    Returns:
        A new MuyGPs model whose specified hyperparameters have been optimized.
    """
    kernel_fn = muygps.kernel.get_opt_fn()
    mean_fn = muygps.get_opt_mean_fn()
    var_fn = muygps.get_opt_var_fn()
    sigma_sq_fn = make_sigma_sq_optim(sigma_method, muygps)

    obj_fn = make_obj_fn(
        obj_method,
        loss_fn,
        kernel_fn,
        mean_fn,
        var_fn,
        sigma_sq_fn,
        pairwise_diffs,
        crosswise_diffs,
        batch_nn_targets,
        batch_targets,
        batch_features=batch_features,
        loss_kwargs=loss_kwargs,
    )

    return _switch_on_opt_method(
        opt_method,
        _bayes_opt_optimize,
        _scipy_optimize,
        muygps,
        obj_fn,
        verbose=verbose,
        **kwargs,
    )
