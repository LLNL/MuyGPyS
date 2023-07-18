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


from bayes_opt import BayesianOptimization
from typing import Dict, Optional

import MuyGPyS._src.math as mm
from MuyGPyS._src.optimize.chassis import (
    _scipy_optimize,
    _bayes_opt_optimize,
)
from MuyGPyS._src.optimize.chassis.numpy import (
    _new_muygps,
    _get_opt_lists,
    _bayes_get_kwargs,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.tensors import make_train_tensors
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.utils import _switch_on_opt_method
from MuyGPyS.optimize.objective import make_obj_fn
from MuyGPyS.optimize.loss import get_loss_func
from MuyGPyS.optimize.sigma_sq import (
    make_sigma_sq_optim,
    muygps_sigma_sq_optim,
)


def optimize_from_tensors(
    muygps: MuyGPS,
    batch_targets: mm.ndarray,
    batch_nn_targets: mm.ndarray,
    crosswise_diffs: mm.ndarray,
    pairwise_diffs: mm.ndarray,
    batch_features: Optional[mm.ndarray] = None,
    loss_method: str = "mse",
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
            Matrix of floats of shape `(batch_count, response_count)` whose
            rows give the expected response for each batch element.
        batch_nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, response_count)`
            containing the expected response for each nearest neighbor of each
            batch element.
        crosswise_diffs:
            A tensor of shape `(batch_count, nn_count, feature_count)` whose
            last two dimensions list the difference between each feature of
            each batch element element and its nearest neighbors.
        pairwise_diffs:
            A tensor of shape `(batch_count, nn_count, nn_count,
            feature_count)` containing the `(nn_count, nn_count,
            feature_count)`-shaped pairwise nearest neighbor difference
            tensors corresponding to each of the batch elements.
        loss_method:
            Indicates the loss function to be used.
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
    loss_fn = get_loss_func(loss_method)
    kernel_fn = muygps.kernel.get_opt_fn()
    mean_fn = muygps.get_opt_mean_fn()
    var_fn = muygps.get_opt_var_fn()
    sigma_sq_fn = make_sigma_sq_optim(sigma_method, muygps)

    obj_fn = make_obj_fn(
        obj_method,
        loss_method,
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


def optimize_from_tensors_mini_batch(
    muygps: MuyGPS,
    train_features: mm.ndarray,
    train_responses: mm.ndarray,
    nbrs_lookup: NN_Wrapper,
    batch_count: int,
    train_count: int,
    num_epochs: int = 1,
    batch_features: Optional[mm.ndarray] = None,
    loss_method: str = "mse",
    obj_method: str = "loo_crossval",
    sigma_method: Optional[str] = "analytic",
    loss_kwargs: Dict = dict(),
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    """
    Find the optimal model using:
    1. Nearest neighbor distance difference matrices
    2. Bayes Optimization
    3. numpy math backend

    See the following example, where we have already constructed exact
    or approximate KNN data lookups a `nbrs_lookup` data structure using
    :class:`MuyGPyS.neighbors.NN_Wrapper`, initialized a
    :class:`~MuyGPyS.gp.muygps.MuyGPS` model `muygps`, created a
    :class:`utils.UnivariateSampler` instance `sampler`.

    Example:
        >>> batch_count=100
        >>> train_count=sampler.train_count
        >>> num_epochs=int(sampler.train_count / batch_count)
        >>> from MuyGPyS.optimize.chassis import
        >>>     optimize_from_tensors_mini_batch
        >>> muygps = optimize_from_tensors_mini_batch(
        ...     muygps,
        ...     train_features,
        ...     train_responses,
        ...     nbrs_lookup,
        ...     batch_count=batch_count,
        ...     train_count=train_count,
        ...     num_epochs=num_epochs,
        ...     batch_features=None,
        ...     loss_method='lool',
        ...     obj_method='loo_crossval',
        ...     sigma_method='analytic',
        ...     verbose=True,
        ...     random_state=1,
        ...     init_points=5,
        ...     n_iter=20,
        ... )
        parameters to be optimized: ['nu']
        bounds: [[0.1 5. ]]
        initial x0: [0.49355858]
        |   iter    |  target   |    nu     |
        -------------------------------------
        | 1         | 538.9     | 0.4936    |
        | 2         | 1.063e+03 | 2.143     |
        | 3         | 726.4     | 3.63      |
        | 4         | 237.9     | 0.1006    |
        | 5         | 1.06e+03  | 1.581     |
        | 6         | 732.3     | 0.8191    |
        | 7         | 362.2     | 5.0       |
        | 8         | 945.4     | 2.772     |
        | 9         | 1.088e+03 | 1.856     |
        | 10        | 1.085e+03 | 1.772     |
        | 11        | 1.087e+03 | 1.907     |
        | 12        | 1.088e+03 | 1.848     |
        | 13        | 1.088e+03 | 1.849     |
        | 14        | 1.088e+03 | 1.85      |
        | 15        | 1.088e+03 | 1.85      |
        | 16        | 1.088e+03 | 1.85      |
        | 17        | 1.088e+03 | 1.851     |
        | 18        | 1.088e+03 | 1.851     |
        | 19        | 1.088e+03 | 1.852     |
        | 20        | 1.088e+03 | 1.852     |
        ...
        epoch	probe point
        -----	-----------
        0, 0.4935585846939505
        1, 1.8520552693068661
        2, 1.862615982964366
        3, 1.8596379807155798
        4, 1.8104353512478297

    Args:
        muygps:
            The model to be optimized.
        train_features:
            Explanatory variables used to train model.
        train_responses:
            Labels corresponding to features used to train model.
        nbrs_lookup:
            Trained nearest neighbor query data structure.
        batch_count:
            The number of batch elements to sample.
        train_count:
            The total number of training examples.
        num_epochs:
            The number of iterations for optimization loop.
        batch_features:
            Set to None, ignore hierarchical stuff for now.
        loss_method:
            Indicates the loss function to be used.
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
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

    # Get objective function components
    loss_fn = get_loss_func(loss_method)
    kernel_fn = muygps.kernel.get_opt_fn()
    mean_fn = muygps.get_opt_mean_fn()
    var_fn = muygps.get_opt_var_fn()
    sigma_sq_fn = make_sigma_sq_optim(sigma_method, muygps)

    # Create bayes_opt kwargs
    x0_names, x0, bounds = _get_opt_lists(muygps, verbose=verbose)
    x0_map = {n: x0[i] for i, n in enumerate(x0_names)}
    bounds_map = {n: bounds[i] for i, n in enumerate(x0_names)}
    optimizer_kwargs, maximize_kwargs = _bayes_get_kwargs(
        verbose=verbose,
        **kwargs,
    )
    if num_epochs > 1:
        optimizer_kwargs["allow_duplicate_points"] = True
    if "init_points" not in maximize_kwargs:
        maximize_kwargs["init_points"] = 5
    if "n_iter" not in maximize_kwargs:
        maximize_kwargs["n_iter"] = 20

    # Initialize list of points to probe
    to_probe = [x0_map]

    # Run optimization loop
    for epoch in range(num_epochs):
        # Sample a batch of points
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )

        # Coalesce distance and target tensors
        (
            batch_crosswise_diffs,
            batch_pairwise_diffs,
            batch_targets,
            batch_nn_targets,
        ) = make_train_tensors(
            batch_indices,
            batch_nn_indices,
            train_features,
            train_responses,
        )

        # Generate the objective function
        obj_fn = make_obj_fn(
            obj_method,
            loss_method,
            loss_fn,
            kernel_fn,
            mean_fn,
            var_fn,
            sigma_sq_fn,
            batch_pairwise_diffs,
            batch_crosswise_diffs,
            batch_nn_targets,
            batch_targets,
            batch_features=batch_features,
            loss_kwargs=loss_kwargs,
        )

        # Create the Bayes optimizer
        optimizer = BayesianOptimization(
            f=obj_fn,
            pbounds=bounds_map,
            **optimizer_kwargs,
        )

        # Probe all explored points
        for point in to_probe:
            optimizer.probe(point, lazy=True)

        # Find maximum of the acquisition function
        optimizer.maximize(**maximize_kwargs)

        # Add explored points to be probed
        to_probe.append(optimizer.max["params"])

    # Print max param per epoch
    if verbose:
        print("\nepoch\tprobe point\n-----\t-----------")
        for epoch in range(num_epochs):
            print(f"{epoch}, {to_probe[epoch].get('nu')}")

    # Compute optimal variance scaling hyperparameter
    ret = muygps_sigma_sq_optim(
        _new_muygps(muygps, x0_names, bounds, optimizer.max["params"]),
        batch_pairwise_diffs,
        batch_nn_targets,
        sigma_method=sigma_method,
    )

    return ret
