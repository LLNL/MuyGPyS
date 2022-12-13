# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
from MuyGPyS import config

if config.muygpys_torch_enabled is False:
    config.update("muygpys_torch_enabled", True)

if config.muygpys_jax_enabled is True:
    config.update("muygpys_jax_enabled", False)

import numpy as np
import torch


from MuyGPyS.gp.distance import pairwise_distances, crosswise_distances
from MuyGPyS._src.optimize.sigma_sq import _analytic_sigma_sq_optim

from MuyGPyS._src.gp.muygps import (
    _muygps_compute_solve,
    _muygps_compute_diagonal_variance,
)
from MuyGPyS.pytorch.muygps_layer import kernel_func


def predict_single_model(
    model,
    test_features,
    train_features,
    train_responses,
    nbrs_lookup,
    nn_count,
    variance_mode="diagonal",
    apply_sigma_sq=True,
):
    """
    Generate predictions using a PyTorch model containing at least one
    MuyGPs_layer in its structure.

    Args:
        model:
            A custom PyTorch.nn.Module object containing at least one
            MuyGPs_layer.
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

    Returns:
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

    train_features_embedded = model.embedding(train_features).detach().numpy()
    test_features_embedded = model.embedding(test_features).detach().numpy()

    nn_indices_test, _ = nbrs_lookup._get_nns(
        test_features_embedded, nn_count=nn_count
    )

    nn_indices_test = torch.from_numpy(nn_indices_test.astype(np.int64))

    train_features_embedded = torch.from_numpy(train_features_embedded).float()
    test_features_embedded = torch.from_numpy(test_features_embedded).float()

    test_nn_targets = torch.from_numpy(
        train_responses[nn_indices_test, :]
    ).float()

    test_count = test_features_embedded.shape[0]

    crosswise_dists = crosswise_distances(
        test_features_embedded,
        train_features_embedded,
        torch.arange(test_count),
        nn_indices_test,
        metric="l2",
    )

    pairwise_dists = pairwise_distances(
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

    predictions = _muygps_compute_solve(K, Kcross, test_nn_targets, model.eps)

    sigma_sq = _analytic_sigma_sq_optim(K, test_nn_targets, model.eps)

    if variance_mode is None:
        return predictions
    elif variance_mode == "diagonal":
        variances = _muygps_compute_diagonal_variance(K, Kcross, model.eps)
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
    num_responses,
    test_features,
    train_features,
    train_responses,
    nbrs_lookup,
    nn_count,
):
    """
    Generate predictions using a PyTorch model containing at least one
    MultivariateMuyGPs_layer in its structure. Meant for the case in which there
    is more than one GP model used to model multiple outputs.

    Args:
        model:
            A custom PyTorch.nn.Module object containing at least one
            MuyGPs_layer.
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

    Returns:
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
        response or a torch.Tensor of shape (response_count,) for a
        multidimensional response. Only returned where apply_sigma_sq is set to
        True.
    """

    train_features_embedded = model.embedding(train_features).detach().numpy()
    test_features_embedded = model.embedding(test_features).detach().numpy()

    nn_indices_test, _ = nbrs_lookup._get_nns(
        test_features_embedded, nn_count=nn_count
    )

    nn_indices_test = torch.from_numpy(nn_indices_test.astype(np.int64))

    train_features_embedded = torch.from_numpy(train_features_embedded).float()
    test_features_embedded = torch.from_numpy(test_features_embedded).float()

    test_nn_targets = torch.from_numpy(
        train_responses[nn_indices_test, :]
    ).float()

    test_count = test_features_embedded.shape[0]

    crosswise_dists = crosswise_distances(
        test_features_embedded,
        train_features_embedded,
        torch.arange(test_count),
        nn_indices_test,
        metric="l2",
    )

    pairwise_dists = pairwise_distances(
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
            K[:, :, :, i],
            Kcross[:, :, i],
            test_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
            model.eps[i],
        ).reshape(batch_count)
        variances[:, i] = _muygps_compute_diagonal_variance(
            K[:, :, :, i], Kcross[:, :, i], model.eps[i]
        )
        sigma_sq[i] = _analytic_sigma_sq_optim(
            K[:, :, :, i],
            test_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
            model.eps[i],
        )
    return predictions, variances, sigma_sq
