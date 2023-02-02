# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Resources and high-level API for a deep kernel learning with MuyGPs.

:func:`~MuyGPyS.examples.muygps_torch.train_deep_kernel_muygps` is a high-level 
API for training deep kernel MuyGPs models for regression.

:func:`~MuyGPyS.examples.muygps_torch.predict_model` is a high-level API for 
generating predictions at test locations given a trained model. 
"""
from MuyGPyS import config
from typing import Dict, List, Optional, Tuple, Union

if config.state.backend != "torch":
    import warnings

    if config.state.torch_enabled == True:
        config.update("muygpys_backend", "torch")
        warnings.warn(
            "Switching to torch backend in order to run torch-only code"
        )
    else:
        warnings.warn(
            f"Bad attempt to execute torch-only code with torch disabled. "
            f"Allowing code to fail."
        )

import numpy as np
import torch

from MuyGPyS.neighbors import NN_Wrapper

from MuyGPyS._src.gp.distance.torch import (
    _pairwise_distances,
    _crosswise_distances,
)
from MuyGPyS._src.optimize.sigma_sq.torch import _analytic_sigma_sq_optim

from MuyGPyS._src.gp.muygps.torch import (
    _muygps_compute_solve,
    _muygps_compute_diagonal_variance,
)
from MuyGPyS._src.gp.noise.torch import _homoscedastic_perturb
from MuyGPyS._src.optimize.loss.torch import _lool_fn as lool_fn


from MuyGPyS.torch.muygps_layer import kernel_func
from torch.optim.lr_scheduler import ExponentialLR

mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
bce_loss = torch.nn.BCELoss()
ce_loss = torch.nn.CrossEntropyLoss()


def predict_single_model(
    model,
    test_features: torch.Tensor,
    train_features: torch.Tensor,
    train_responses: torch.Tensor,
    nbrs_lookup: NN_Wrapper,
    nn_count: torch.int64,
    variance_mode="diagonal",
    apply_sigma_sq=True,
):
    """
    Generate predictions using a PyTorch model containing at least one
    `MuyGPyS.torch.muygps_layer.MuyGPs_layer` in its structure. Note that
    the custom PyTorch MuyGPs_layer objects only support the Matern kernel.
    Support for more kernels will be added in future releases.

    Args:
        model:
            A custom PyTorch.nn.Module object containing an
            `embedding` component and one
            `MuyGPyS.torch.muygps_layer.MuyGPs_layer` layer.
        test_features:
            A torch.Tensor of shape `(test_count, feature_count)` containing
            the test features to be regressed.
        train_features:
            A torch.Tensor of shape `(train_count, feature_count)` containing
            the training features.
        train_responses:
            A torch.Tensor of shape `(train_count, response_count)` containing
            the training responses corresponding to each feature.
        nbrs_lookup:
            A NN_Wrapper nearest neighbor lookup data structure.
        variance_mode:
            Specifies the type of variance to return. Currently supports
            `"diagonal"` and None. If None, report no variance term.
        apply_sigma_sq:
            Indicates whether to scale the posterior variance by `sigma_sq`.
            Unused if `variance_mode is None` or if set to False`.

    Returns
    -------
    predictions:
        A torch.Tensor of shape `(test_count, response_count)` whose rows are
        the predicted response for each of the given test feature.
    variances:
        A torch.Tensor of shape `(batch_count,)` consisting of the diagonal
        elements of the posterior variance, or a matrix of shape
        `(batch_count, response_count)` for a multidimensional response.
        Only returned where `variance_mode == "diagonal"`.
    sigma_sq:
        A scalar used to rescale the posterior variance if a univariate
        response or a torch.Tensor of shape `(response_count,)` for a
        multidimensional response. Only returned where apply_sigma_sq is set to
        True.
    """
    if model.embedding is None:
        raise NotImplementedError(f"MuyGPs PyTorch model requires embedding.")
    train_features_embedded = model.embedding(train_features).detach().numpy()
    test_features_embedded = model.embedding(test_features).detach().numpy()

    test_count = test_features_embedded.shape[0]

    nn_indices_test, _ = nbrs_lookup._get_nns(
        test_features_embedded, nn_count=nn_count
    )

    nn_indices_test = torch.from_numpy(nn_indices_test.astype(np.int64))

    train_features_embedded = torch.from_numpy(train_features_embedded).float()
    test_features_embedded = torch.from_numpy(test_features_embedded).float()

    test_nn_targets = train_responses[nn_indices_test, :]

    crosswise_dists = _crosswise_distances(
        test_features_embedded,
        train_features_embedded,
        torch.arange(test_count),
        nn_indices_test,
        metric="l2",
    )

    pairwise_dists = _pairwise_distances(
        train_features_embedded, nn_indices_test, metric="l2"
    )

    Kcross = kernel_func(
        crosswise_dists,
        nu=model.nu,
        length_scale=model.length_scale,
    )
    K = kernel_func(
        pairwise_dists,
        nu=model.nu,
        length_scale=model.length_scale,
    )

    predictions = _muygps_compute_solve(
        _homoscedastic_perturb(K, model.eps), Kcross, test_nn_targets
    )

    sigma_sq = _analytic_sigma_sq_optim(
        _homoscedastic_perturb(K, model.eps), test_nn_targets
    )

    if variance_mode is None:
        return predictions
    elif variance_mode == "diagonal":
        variances = _muygps_compute_diagonal_variance(
            _homoscedastic_perturb(K, model.eps), Kcross
        )
        if apply_sigma_sq is True:
            if len(sigma_sq) == 1:
                variances *= sigma_sq
            else:
                variances = torch.outer(variances, sigma_sq)
    else:
        raise NotImplementedError(
            f"Variance mode {variance_mode} is not implemented."
        )

    return predictions, variances, sigma_sq


def predict_multiple_model(
    model,
    num_responses: torch.int64,
    test_features: torch.Tensor,
    train_features: torch.Tensor,
    train_responses: torch.Tensor,
    nbrs_lookup: torch.int64,
    nn_count: torch.int64,
    variance_mode="diagonal",
    apply_sigma_sq=True,
):
    """
    Generate predictions using a PyTorch model containing a
    `MuyGPyS.torch.muygps_layer.MultivariateMuyGPs_layer` in its structure.
    Meant for the case in which there is more than one GP model used to model
    multiple outputs. Note that the custom PyTorch MultivariateMuyGPs_layer
    objects only support the Matern kernel. Support for more kernels will be
    added in future releases.

    Args:
        model:
            A custom PyTorch.nn.Module object containing an
            `embedding` component and one
            `MuyGPyS.torch.muygps_layer.MultivariateMuyGPs_layer` layer.
        test_features:
            A torch.Tensor of shape `(test_count, feature_count)` containing
            the test features to be regressed.
        train_features:
            A torch.Tensor of shape `(train_count, feature_count)` containing
            the training features.
        train_responses:
            A torch.Tensor of shape `(train_count, response_count)` containing
            the training responses corresponding to each feature.
        nbrs_lookup:
            A NN_Wrapper nearest neighbor lookup data structure.
        variance_mode:
            Specifies the type of variance to return. Currently supports
            `"diagonal"` and None. If None, report no variance term.
        apply_sigma_sq:
            Indicates whether to scale the posterior variance by `sigma_sq`.
            Unused if `variance_mode is None` or if set to False`.

    Returns
    -------
    predictions:
        A torch.Tensor of shape `(test_count, response_count)` whose rows are
        the predicted response for each of the given test feature.
    variances:
        A torch.Tensor of shape `(batch_count,)` consisting of the diagonal
        elements of the posterior variance, or a matrix of shape
        `(batch_count, response_count)` for a multidimensional response.
        Only returned where `variance_mode == "diagonal"`.
    sigma_sq:
        A scalar used to rescale the posterior variance if a univariate
        response or a torch.Tensor of shape `(response_count,)` for a
        multidimensional response. Only returned where apply_sigma_sq is set to
        True.
    """
    if model.embedding is None:
        raise NotImplementedError(f"MuyGPs PyTorch model requires embedding.")

    train_features_embedded = model.embedding(train_features).detach().numpy()
    test_features_embedded = model.embedding(test_features).detach().numpy()

    test_count = test_features_embedded.shape[0]

    nn_indices_test, _ = nbrs_lookup._get_nns(
        test_features_embedded, nn_count=nn_count
    )

    nn_indices_test = torch.from_numpy(nn_indices_test.astype(np.int64))

    train_features_embedded = torch.from_numpy(train_features_embedded).float()
    test_features_embedded = torch.from_numpy(test_features_embedded).float()

    test_nn_targets = train_responses[nn_indices_test, :]

    crosswise_dists = _crosswise_distances(
        test_features_embedded,
        train_features_embedded,
        torch.arange(test_count),
        nn_indices_test,
        metric="l2",
    )

    pairwise_dists = _pairwise_distances(
        train_features_embedded, nn_indices_test, metric="l2"
    )

    (
        batch_count,
        nn_count,
        response_count,
    ) = model.batch_nn_targets.shape

    Kcross = torch.zeros(test_count, nn_count, response_count)
    K = torch.zeros(test_count, nn_count, nn_count, response_count)

    for i in range(num_responses):
        Kcross[:, :, i] = kernel_func(
            crosswise_dists,
            nu=model.nu[i],
            length_scale=model.length_scale[i],
        )

        K[:, :, :, i] = kernel_func(
            pairwise_dists,
            nu=model.nu[i],
            length_scale=model.length_scale[i],
        )

    batch_count, nn_count, response_count = test_nn_targets.shape

    predictions = torch.zeros(batch_count, response_count)
    variances = torch.zeros(batch_count, response_count)
    sigma_sq = torch.zeros(
        response_count,
    )

    for i in range(model.num_models):
        predictions[:, i] = _muygps_compute_solve(
            _homoscedastic_perturb(K[:, :, :, i], model.eps[i]),
            Kcross[:, :, i],
            test_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
        ).reshape(batch_count)
        variances[:, i] = _muygps_compute_diagonal_variance(
            _homoscedastic_perturb(K[:, :, :, i], model.eps[i]),
            Kcross[:, :, i],
        )
        sigma_sq[i] = _analytic_sigma_sq_optim(
            _homoscedastic_perturb(K[:, :, :, i], model.eps[i]),
            test_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
        )
    return predictions, variances, sigma_sq


def predict_model(
    model,
    test_features: torch.Tensor,
    train_features: torch.Tensor,
    train_responses: torch.Tensor,
    nbrs_lookup: NN_Wrapper,
    nn_count: torch.int64,
    variance_mode="diagonal",
    apply_sigma_sq=True,
):
    """
    Generate predictions using a PyTorch model containing a
    `MuyGPyS.torch.muygps_layer.MuyGPs_layer` layer or a
    `MuyGPyS.torch.muygps_layer.MultivariateMuyGPs_layer` layer in its
    structure. Note that the custom PyTorch layers for MuyGPs
    objects only support the Matern kernel. Support for more kernels will be
    added in future releases.

    Example:
        >>> #model must be defined as a PyTorch model inheriting from
        ... #torch.nn.Module. Must have two components: model.embedding
        ... #(e.g., a neural net) and another component model.GP_layer.
        >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
        >>> from MuyGPyS.neighbors import NN_Wrapper
        >>> train, test = _make_gaussian_data(10000, 1000, 100, 10)
        >>> nn_count = 10
        >>> nbrs_lookup = NN_Wrapper(train['input'], nn_count, nn_method="hnsw")
        >>> predictions, variances, sigma_sq = predict_model(
        ... model,
        ... torch.from_numpy(test['input']),
        ... torch.from_numpy(train['input']),
        ... torch.from_numpy(train['output']),
        ... nbrs_lookup,
        ... nn_count)



    Args:
        model:
            A custom PyTorch.nn.Module object containing an embedding component
            and one MuyGPs_layer or MultivariateMuyGPS_layer layer.
        test_features:
            A torch.Tensor of shape `(test_count, feature_count)` containing
            the test features to be regressed.
        train_features:
            A torch.Tensor of shape `(train_count, feature_count)` containing
            the training features.
        train_responses:
            A torch.Tensor of shape `(train_count, response_count)` containing
            the training responses corresponding to each feature.
        nbrs_lookup:
            A NN_Wrapper nearest neighbor lookup data structure.
        variance_mode:
            Specifies the type of variance to return. Currently supports
            `"diagonal"` and None. If None, report no variance term.
        apply_sigma_sq:
            Indicates whether to scale the posterior variance by `sigma_sq`.
            Unused if `variance_mode is None` or if set to False`.

    Returns
    -------
    predictions:
        A torch.Tensor of shape `(test_count, response_count)` whose rows are
        the predicted response for each of the given test feature.
    variances:
        A torch.Tensor of shape `(batch_count,)` consisting of the diagonal
        elements of the posterior variance, or a matrix of shape
        `(batch_count, response_count)` for a multidimensional response.
        Only returned where `variance_mode == "diagonal"`.
    sigma_sq:
        A scalar used to rescale the posterior variance if a univariate
        response or a torch.Tensor of shape `(response_count,)` for a
        multidimensional response. Only returned where apply_sigma_sq is set to
        True.
    """
    if model.GP_layer is None:
        raise NotImplementedError(f"MuyGPs PyTorch model requires GP_layer.")
    if hasattr(model.GP_layer, "num_models"):
        return predict_multiple_model(
            model,
            model.GP_layer.num_models,
            test_features,
            train_features,
            train_responses,
            nbrs_lookup,
            nn_count,
            variance_mode="diagonal",
            apply_sigma_sq=True,
        )
    else:
        return predict_single_model(
            model,
            test_features,
            train_features,
            train_responses,
            nbrs_lookup,
            nn_count,
            variance_mode="diagonal",
            apply_sigma_sq=True,
        )


def train_deep_kernel_muygps(
    model,
    train_features: torch.Tensor,
    train_responses: torch.Tensor,
    batch_indices: torch.Tensor,
    nbrs_lookup: NN_Wrapper,
    training_iterations=10,
    optimizer_method=torch.optim.Adam,
    learning_rate=1e-3,
    scheduler_decay=0.95,
    loss_function="lool",
    update_frequency=1,
    verbose=False,
    nn_kwargs: Dict = dict(),
):
    """
    Train a PyTorch model containing an embedding component and
    a `MuyGPyS.torch.muygps_layer.MuyGPs_layer` layer or a
    `MuyGPyS.torch.muygps_layer. MultivariateMuyGPs_layer` layer in its
    structure. Note that the custom PyTorch layers for MuyGPs models only
    support the Matern kernel. Support for more kernels will be added in
    future releases.

    Example:
        >>> #model must be defined as a PyTorch model inheriting from
        ... #torch.nn.Module. Must have two components: model.embedding
        ... #(e.g., a neural net) and another component model.GP_layer.
        >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
        >>> from MuyGPyS.neighbors import NN_Wrapper
        >>> from MuyGPyS.examples.muygps_torch import train_deep_kernel_muygps
        >>> from MuyGPyS._src.optimize.loss import _lool_fn as lool_fn
        >>> train, test = _make_gaussian_data(10000, 1000, 100, 10)
        >>> nn_count = 10
        >>> nbrs_lookup = NN_Wrapper(train['input'], nn_count, nn_method="hnsw")
        >>> batch_count = 100
        >>> train_count = 10000
        >>> batch_indices, batch_nn_indices = sample_batch(nbrs_lookup, batch_count, train_count)
        >>> nbrs_struct, model_trained = train_deep_kernel_muygps(
        ... model=model,
        ... train_features=torch.from_numpy(train['input']),
        ... train_responses=torch.from_numpy(train['output']),
        ... batch_indices=torch.from_numpy(batch_indices),
        ... nbrs_lookup=nbrs_lookup,
        ... training_iterations=10,
        ... optimizer_method=torch.optim.Adam,
        ... learning_rate=1e-3,
        ... scheduler_decay=0.95,
        ... loss_function=lool_fn,
        ... update_frequency=1)

    Args:
        model:
            A custom PyTorch.nn.Module object containing at least one
            embedding layer and one MuyGPs_layer or MultivariateMuyGPS_layer
            layer.
        train_features:
            A torch.Tensor of shape `(train_count, feature_count)` containing
            the training features.
        train_responses:
            A torch.Tensor of shape `(train_count, response_count)` containing
            the training responses corresponding to each feature.
        batch_indices:
            A torch.Tensor of shape `(batch_count,)` containing the indices of
            the training batch.
        nbrs_lookup:
            A NN_Wrapper nearest neighbor lookup data structure.
        training_iterations:
            The number of training iterations to be used in training.
        optimizer method:
            An optimization method from the torch.optim class.
        learning_rate:
            The learning rate to be applied during training.
        schedule_decay:
            The exponential decay rate to be applied to the learning rate.
        loss function:
            The loss function to be used in training. Defaults to "lool" for
            leave-one-out likelihood. Other options are "mse" for
            mean-squared error, "ce" for cross entropy loss, "bce" for binary
            cross entropy loss, and "l1" for L1 loss.
        update_frequency:
            Tells the training procedure how frequently the nearest neighbor
            structure should be updated. An update frequency of n indicates that
            every n epochs the nearest neighbor structure should be updated.
        verbose:
            Indicates whether or not to include print statements during
            training.
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.

    Returns
    -------
    nbrs_lookup:
        A NN_Wrapper object containing the nearest neighbors of the embedded
        training data.
    model:
        A trained deep kernel MuyGPs model.
    """
    if model.embedding is None:
        raise NotImplementedError(f"MuyGPs PyTorch model requires embedding.")
    optimizer = optimizer_method(
        [
            {"params": model.parameters()},
        ],
        lr=learning_rate,
    )
    scheduler = ExponentialLR(optimizer, gamma=scheduler_decay)
    nn_count = nbrs_lookup.nn_count
    batch_features = train_features[batch_indices, :]
    batch_responses = train_responses[batch_indices, :]

    loss_function = loss_function.lower()

    if loss_function == "mse":
        loss_func = mse_loss
    elif loss_function == "bce":
        loss_func = bce_loss
    elif loss_function == "ce":
        loss_func = ce_loss
    elif loss_function == "lool":
        loss_func = lool_fn
    else:
        raise ValueError(f"loss function {loss_function} is not supported")

    for i in range(training_iterations):
        model.train()
        optimizer.zero_grad()
        predictions, variances, sigma_sq = model(train_features)

        if loss_function == "lool":
            loss = loss_func(
                predictions.squeeze(),
                batch_responses.squeeze(),
                variances.squeeze(),
                sigma_sq.squeeze(),
            )
        else:
            loss = loss_func(predictions, batch_responses)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if np.mod(i, update_frequency) == 0:
            if verbose == True:
                print(
                    "Iter %d/%d - Loss: %.10f"
                    % (i + 1, training_iterations, loss.item())
                )
            model.eval()
            nbrs_lookup = NN_Wrapper(
                model.embedding(train_features).detach().numpy(),
                nn_count,
                **nn_kwargs,
            )
            batch_nn_indices, _ = nbrs_lookup._get_nns(
                model.embedding(batch_features).detach().numpy(),
                nn_count=nn_count,
            )
            batch_nn_indices = torch.from_numpy(
                batch_nn_indices.astype(np.int64)
            )
            batch_nn_targets = train_responses[batch_nn_indices, :]

            model.batch_nn_indices = batch_nn_indices
            model.batch_nn_targets = batch_nn_targets

        torch.cuda.empty_cache()
    nbrs_lookup = NN_Wrapper(
        model.embedding(train_features).detach().numpy(),
        nn_count,
        **nn_kwargs,
    )
    batch_nn_indices, _ = nbrs_lookup._get_nns(
        model.embedding(batch_features).detach().numpy(), nn_count=nn_count
    )
    batch_nn_indices = torch.from_numpy(batch_nn_indices.astype(np.int64))
    batch_nn_targets = train_responses[batch_nn_indices, :]
    model.batch_nn_indices = batch_nn_indices
    model.batch_nn_targets = batch_nn_targets
    return nbrs_lookup, model


def update_nearest_neighbors(
    model,
    train_features: torch.Tensor,
    train_responses: torch.Tensor,
    batch_indices: torch.Tensor,
    nn_count: torch.int64,
    nn_kwargs: Dict = dict(),
):
    """
    Update the nearest neighbors after deformation via a PyTorch model
    containing an embedding component and a
    `MuyGPyS.torch.muygps_layer.MuyGPs_layer` layer or a
    `MuyGPyS.torch.muygps_layer. MultivariateMuyGPs_layer` layer in its
    structure.

    Example:
        >>> #model must be defined as a PyTorch model inheriting from
        ... #torch.nn.Module. Must have two components: model.embedding
        ... #(e.g., a neural net) and another component model.GP_layer.
        >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
        >>> from MuyGPyS.neighbors import NN_Wrapper
        >>> from MuyGPyS.examples.muygps_torch import update_nearest_neighbors
        >>> train, test = _make_gaussian_data(10000, 1000, 100, 10)
        >>> nn_count = 10
        >>> batch_count = 100
        >>> train_count = 10000
        >>> batch_indices, batch_nn_indices = sample_batch(nbrs_lookup, batch_count, train_count)
        >>> nbrs_struct, model_trained = update_nearest_neighbors(
        ... model=model,
        ... train_features=torch.from_numpy(train['input']),
        ... train_responses=torch.from_numpy(train['output']),
        ... batch_indices=torch.from_numpy(batch_indices),
        ... nn_count=nn_count,)

    Args:
        model:
            A custom PyTorch.nn.Module object containing at least one
            embedding layer and one MuyGPs_layer or MultivariateMuyGPS_layer
            layer.
        train_features:
            A torch.Tensor of shape `(train_count, feature_count)` containing
            the training features.
        train_responses:
            A torch.Tensor of shape `(train_count, response_count)` containing
            the training responses corresponding to each feature.
        batch_indices:
            A torch.Tensor of shape `(batch_count,)` containing the indices of
            the training batch.
        nn_count:
            A torch.int64 giving the number of nearest neighbors.
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.

    Returns
    -------
    nbrs_lookup:
        A NN_Wrapper object containing the updated nearest neighbors of the
        embedded training data.
    model:
        A deep kernel MuyGPs model with updated nearest neighbors.
    """
    if model.embedding is None:
        raise NotImplementedError(f"MuyGPs PyTorch model requires embedding.")
    batch_features = train_features[batch_indices, :]
    nbrs_lookup = NN_Wrapper(
        model.embedding(train_features).detach().numpy(),
        nn_count,
        **nn_kwargs,
    )
    batch_nn_indices, _ = nbrs_lookup._get_nns(
        model.embedding(batch_features).detach().numpy(), nn_count=nn_count
    )
    batch_nn_indices = torch.from_numpy(batch_nn_indices.astype(np.int64))
    batch_nn_targets = train_responses[batch_nn_indices, :]
    model.batch_nn_indices = batch_nn_indices
    model.batch_nn_targets = batch_nn_targets
    return nbrs_lookup, model
