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
from typing import Dict, Optional, Tuple

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.optimize.chassis.numpy import (
    _new_muygps,
    _get_opt_lists,
    _bayes_get_kwargs,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.distortion import AnisotropicDistortion
from MuyGPyS.gp.tensors import make_train_tensors
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.objective import make_obj_fn
from MuyGPyS.optimize.loss import get_loss_func
from MuyGPyS.optimize.sigma_sq import (
    make_sigma_sq_optim,
    muygps_sigma_sq_optim,
)


def optimize_from_tensors_mini_batch(
    muygps: MuyGPS,
    train_features: mm.ndarray,
    train_responses: mm.ndarray,
    nn_count: int,
    batch_count: int,
    train_count: int,
    num_epochs: int = 1,
    keep_state: bool = False,
    probe_previous: bool = False,
    batch_features: Optional[mm.ndarray] = None,
    loss_method: str = "lool",
    obj_method: str = "loo_crossval",
    sigma_method: Optional[str] = "analytic",
    loss_kwargs: Dict = dict(),
    verbose: bool = False,
    **kwargs,
) -> Tuple[MuyGPS, NN_Wrapper, mm.ndarray]:
    """
    Find the optimal model using:
    1. scikit learn Nearest Neighbors
    2. Bayes Optimization
    3. numpy math backend

    See the following example, where we have already initialized a
    :class:`~MuyGPyS.gp.muygps.MuyGPS` model `muygps` and created a
    :class:`utils.UnivariateSampler` or :class:`utils.UnivariateSampler2D`
    instance `sampler`.

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
        ...     num_epochs: int = 1,
        ...     keep_state: bool = False,
        ...     probe_previous: bool = True,
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
        0, {'nu': 0.4935585846939505}
        1, {'nu': 2.0153664487151066}
        2, {'nu': 1.9821713002046035}
        3, {'nu': 1.9744962201179712}
        4, {'nu': 2.048927092852563}

    Args:
        muygps:
            The model to be optimized.
        train_features:
            Explanatory variables used to train model.
        train_responses:
            Labels corresponding to features used to train model.
        nn_count:
            The number of nearest neighbors to return in queries.
        batch_count:
            The number of batch elements to sample.
        train_count:
            The total number of training examples.
        num_epochs:
            The number of iterations for optimization loop.
        keep_state:
            If True, maintain optimizer target space and explored points.
        probe_previous:
            If True, store max params to probe in next loop iteration.
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
            Additional keyword arguments to be passed to the wrapper
            optimizer.

    Returns:
        A new MuyGPs model whose specified hyperparameters have been optimized.

    Returns
    -------
        MuyGPS:
            A new MuyGPs model with optimized hyperparameters.
        NN_Wrapper:
            Trained nearest neighbor query data structure.
        mm.ndarray:
            The full training data of shape `(train_count, feature_count)` that
            will construct the nearest neighbor query datastructure.
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
    is_anisotropic = False
    if isinstance(muygps.kernel.distortion_fn, AnisotropicDistortion):
        is_anisotropic = True

    # Create Bayes optimizer
    optimizer = BayesianOptimization(
        f=None,
        pbounds=bounds_map,
        **optimizer_kwargs,
    )

    # Initialize nearest neighbors lookup
    nn_count = 30
    nbrs_lookup = NN_Wrapper(
        train_features, nn_count, nn_method="exact", algorithm="ball_tree"
    )
    # train_features_scaled = np.copy(train_features)  # TODO test config #1,#2

    # Sample a batch of points
    if keep_state:
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )
    new_nbrs_lookup = nbrs_lookup

    # Run optimization loop
    to_probe = [x0_map]
    for epoch in range(num_epochs):
        # Sample a batch of points
        if not keep_state:
            batch_indices, batch_nn_indices = sample_batch(
                new_nbrs_lookup, batch_count, train_count
            )

        # Coalesce distance and target tensors
        train_features_scaled = np.copy(train_features)  # TODO test config #3
        (
            batch_crosswise_diffs,
            batch_pairwise_diffs,
            batch_targets,
            batch_nn_targets,
        ) = make_train_tensors(
            batch_indices,
            batch_nn_indices,
            train_features_scaled,
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

        # Setting the function to be optimized for this epoch
        if keep_state:
            optimizer._space.target_func = obj_fn
        else:
            optimizer = BayesianOptimization(
                f=obj_fn,
                pbounds=bounds_map,
                **optimizer_kwargs,
            )

        # Probe explored points
        if probe_previous:
            for point in to_probe:
                optimizer.probe(point, lazy=True)
        elif epoch == 0:
            optimizer.probe(to_probe[0], lazy=True)

        # Find maximum of the acquisition function
        optimizer.maximize(**maximize_kwargs)

        # Add explored points to be probed
        to_probe.append(optimizer.max["params"])
        if verbose:
            print(f"{epoch}, {optimizer.max['params']}")

        # Update neighborhoods using the learned length scales
        if is_anisotropic and (epoch < (num_epochs - 1)):
            length_scales = AnisotropicDistortion._get_length_scale_array(
                mm.array,
                train_features.shape,
                None,
                **optimizer.max["params"],
            )
            # train_features_scaled = ( # TODO test config #1
            #     train_features_scaled / length_scales
            # )
            train_features_scaled = (
                train_features / length_scales  # TODO test config #2,#3
            )
            new_nbrs_lookup = NN_Wrapper(
                train_features_scaled,
                nn_count,
                nn_method="exact",
                algorithm="ball_tree",
            )

    # Compute optimal variance scaling hyperparameter
    new_muygpys = muygps_sigma_sq_optim(
        _new_muygps(muygps, x0_names, bounds, optimizer.max["params"]),
        batch_pairwise_diffs,
        batch_nn_targets,
        sigma_method=sigma_method,
    )

    return new_muygpys, new_nbrs_lookup, train_features_scaled  # TODO
