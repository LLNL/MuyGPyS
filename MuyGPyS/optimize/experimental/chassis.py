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

from copy import deepcopy
import numpy as np
from time import process_time
from typing import Dict, Optional, Tuple

from bayes_opt import BayesianOptimization
import MuyGPyS._src.math as mm
from MuyGPyS._src.optimize.chassis.numpy import (
    _new_muygps,
    _get_opt_lists,
    _bayes_get_kwargs,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import Anisotropy
from MuyGPyS.gp.tensors import make_train_tensors
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.objective import make_loo_crossval_fn
from MuyGPyS.optimize.loss import LossFn, lool_fn


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
    loss_fn: LossFn = lool_fn,
    obj_method: str = "loo_crossval",
    loss_kwargs: Dict = dict(),
    verbose: bool = False,
    **kwargs,
) -> Tuple[MuyGPS, NN_Wrapper, float, int, int]:
    """
    Find the optimal model using:
    1. Exact KNN using scikit learn
    2. Bayes Optimization
    3. numpy math backend

    See the following example, where we have already initialized a
    :class:`~MuyGPyS.gp.muygps.MuyGPS` model `muygps` and created a
    :class:`utils.UnivariateSampler` or :class:`utils.UnivariateSampler2D`
    instance `sampler`.

    Example:
        >>> (
        >>>     muygps_optloop,
        >>>     nbrs_lookup_final,
        >>>     exec_time,
        >>>     probe_count,
        >>>     opt_steps,
        >>> ) = optimize_from_tensors_mini_batch(
        ...     muygps,
        ...     train_features,
        ...     train_responses,
        ...     nn_count=30,
        ...     batch_count=sampler.train_count,
        ...     train_count=sampler.train_count,
        ...     num_epochs=1,
        ...     keep_state=False,
        ...     probe_previous=False,
        ...     loss_fn=lool_fn,
        ...     obj_method="loo_crossval",
        ...     verbose=True,
        ...     random_state=1,
        ...     init_points=5,
        ...     n_iter=20,
        ...     allow_duplicate_points=True,
        ... )
        parameters to be optimized: ['length_scale0', 'length_scale1']
        bounds: [[0.01 1.  ]
        [0.01 1.  ]]
        initial x0: [0.09718538 0.42218699]
        |   iter    |  target   | length... | length... |
        -------------------------------------------------
        | 1         | 793.7     | 0.09719   | 0.4222    |
        | 2         | 425.1     | 0.4229    | 0.7231    |
        | 3         | 39.33     | 0.01011   | 0.3093    |
        | 4         | 83.32     | 0.1553    | 0.1014    |
        | 5         | 480.6     | 0.1944    | 0.3521    |
        | 6         | 277.1     | 0.4028    | 0.5434    |
        | 7         | 790.1     | 0.09769   | 0.4123    |
        | 8         | 685.2     | 0.1737    | 0.4721    |
        | 9         | 792.9     | 0.09583   | 0.4147    |
        | 10        | 513.9     | 0.03333   | 0.6004    |
        | 11        | -444.6    | 1.0       | 0.01      |
        | 12        | 803.1     | 0.187     | 0.9781    |
        | 13        | -181.9    | 0.01      | 1.0       |
        | 14        | 707.7     | 0.3298    | 0.9882    |
        | 15        | 234.0     | 0.5577    | 0.705     |
        | 16        | 675.0     | 0.157     | 0.4126    |
        | 17        | 755.4     | 0.2337    | 0.8346    |
        | 18        | 107.2     | 1.0       | 1.0       |
        | 19        | -108.3    | 1.0       | 0.5391    |
        | 20        | -205.5    | 0.5192    | 0.01      |
        | 21        | 411.2     | 0.5939    | 1.0       |
        | 22        | 729.1     | 0.2109    | 0.6687    |
        | 23        | -98.07    | 0.673     | 0.3635    |
        | 24        | 797.6     | 0.1144    | 0.775     |
        | 25        | 761.3     | 0.2273    | 0.8345    |
        | 26        | 147.3     | 0.7809    | 0.845     |
        =================================================

        0, {'length_scale0': 0.1869..., 'length_scale1': 0.9781...}

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
        loss_fn:
            The loss functor used to evaluate model performance.
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
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
        float:
            Total cpu and system time in loop execution.
        int:
            Total points probed (exploration).
        int:
            Total iterations of bayes optimization (exploitation).
    """

    # Get objective function components
    kernel_fn = muygps.kernel.get_opt_fn()
    mean_fn = muygps.get_opt_mean_fn()
    var_fn = muygps.get_opt_var_fn()
    scale_fn = muygps.scale.get_opt_fn(muygps)

    # Create bayes_opt kwargs
    x0_names, x0, bounds = _get_opt_lists(muygps, verbose=verbose)
    x0_map = {n: x0[i] for i, n in enumerate(x0_names)}
    to_probe = [x0_map]
    bounds_map = {n: bounds[i] for i, n in enumerate(x0_names)}
    optimizer_kwargs, maximize_kwargs = _bayes_get_kwargs(
        verbose=verbose,
        **kwargs,
    )
    total_pts_probed = num_epochs * maximize_kwargs["init_points"]
    total_opt_steps = num_epochs * maximize_kwargs["n_iter"]

    # Create Bayes optimizer
    optimizer = BayesianOptimization(
        f=None,
        pbounds=bounds_map,
        **optimizer_kwargs,
    )
    optimized_values = ["\r\n"]

    # Initialize nearest neighbors lookup and get batch indices
    nbrs_lookup = NN_Wrapper(
        train_features, nn_count, nn_method="exact", algorithm="ball_tree"
    )
    new_nbrs_lookup = deepcopy(nbrs_lookup)
    batch_indices = mm.arange(train_count, dtype=mm.itype)

    # Run optimization loop
    time_start = process_time()
    for epoch in range(num_epochs):
        # Get batch nearest neighbors indices
        if not keep_state and train_count > batch_count:
            batch_indices = mm.iarray(
                np.random.choice(train_count, batch_count, replace=False)
            )
        batch_nn_indices, _ = new_nbrs_lookup.get_batch_nns(batch_indices)

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
        obj_fn = make_loo_crossval_fn(
            loss_fn,
            kernel_fn,
            mean_fn,
            var_fn,
            scale_fn,
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
                total_pts_probed += 1
        elif epoch == 0:
            optimizer.probe(to_probe[0], lazy=True)
            total_pts_probed += 1

        # Find maximum of the acquisition function
        optimizer.maximize(**maximize_kwargs)

        # Add explored points to be probed
        to_probe.append(optimizer.max["params"])
        optimized_values.append(f"{epoch}, {optimizer.max['params']}")

        # Update neighborhoods using the learned length scales
        if isinstance(muygps.kernel.deformation, Anisotropy) and (
            epoch < (num_epochs - 1)
        ):
            length_scales = muygps.kernel.deformation._length_scale_array(
                train_features.shape,
                **optimizer.max["params"],
            )
            train_features_scaled = train_features / length_scales
            new_nbrs_lookup = NN_Wrapper(
                train_features_scaled,
                nn_count,
                nn_method="exact",
                algorithm="ball_tree",
            )
    time_stop = process_time()

    # Print outcomes
    if verbose:
        for line in optimized_values:
            print(line)

    # Compute optimal variance scaling hyperparameter
    new_muygpys = _new_muygps(muygps, x0_names, bounds, optimizer.max["params"])

    new_muygpys = new_muygpys.optimize_scale(
        batch_pairwise_diffs, batch_nn_targets
    )

    return (
        new_muygpys,
        nbrs_lookup,
        (time_stop - time_start),
        total_pts_probed,
        total_opt_steps,
    )
