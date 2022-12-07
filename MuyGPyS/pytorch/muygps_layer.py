# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


# TO DO: Add thorough documentation
"""
MuyGPs PyTorch implementation
"""

import torch
from torch import nn
import numpy as np
from MuyGPyS.gp.distance import pairwise_distances, crosswise_distances
from MuyGPyS.gp.kernels import (
    _matern_05_fn as matern_05_fn,
    _matern_15_fn as matern_15_fn,
    _matern_25_fn as matern_25_fn,
    _matern_inf_fn as matern_inf_fn,
    _matern_gen_fn as matern_gen_fn,
)

from typing import Callable
from MuyGPyS.neighbors import NN_Wrapper

from MuyGPyS._src.gp.muygps import (
    _muygps_compute_solve,
    _muygps_compute_diagonal_variance,
)
from MuyGPyS._src.optimize.sigma_sq import _analytic_sigma_sq_optim


class MuyGPs_layer(nn.Module):
    """
    MuyGPs model written as a custom PyTorch layer using nn.Module.

    Implements the MuyGPs algorithm as articulated in [muyskens2021muygps]_. See
    documentation on MuyGPs class for more detail.

    PyTorch does not currently support the Bessel function required to compute
    the Matern kernel for non-special values of :math:`\\nu`, e.g. 1/2, 3/2,
    5/2, and :math:`\\infty`. The MuyGPs layer allows the lengthscale parameter
    :math:`\\rho` to be trained (provided an initial value by the user) as well
    as the homoscedastic :math:`\\varepsilon` noise parameter.

    The MuyGPs layer returns the posterior mean, posterior variance, and a
    vector of :math:`\\sigma^2` indicating the scale parameter associated
    with the posterior variance of each dimension of the response.

    :math:`\\sigma^2` is the only parameter assumed to be a training target by
    default, and is treated differently from all other hyperparameters. All
    other training targets must be manually specified in the construction of
    a MuyGPs_layer object.

    Example:
        >>> from MuyGPyS.pytorch.muygps_layer import MuyGPs_layer
        >>> kernel_eps = 1e-3
        >>> nu = 1/2
        >>> length_scale = 1.0
        >>> batch_indices = torch.arange(100,)
        >>> batch_nn_indices = torch.arange(100,)
        >>> batch_targets = torch.ones(100,)
        >>> batch_nn_targets = torch.ones(100,)
        >>> muygps_layer_object = MuyGPs_layer(kernel_eps, nu, length_scale,
        batch_indices, batch_nn_indices, batch_targets, batch_nn_targets)



    Args:
        kernel_eps:
            A hyperparameter corresponding to the aleatoric uncertainty in the
            data.
        nu:
            A smoothness parameter allowed to take on values 1/2, 3/2, 5/2, or
            torch.inf.
        length_scale:
            The length scale parameter in the Matern kernel.
        batch_indices:
            A torch.Tensor of shape (batch_count,) containing the indices of
            the training data to be sampled for training.
        batch_nn_indices:
            A torch.Tensor of shape (batch_count, nn_count) containing the
            indices of the k nearest neighbors of the batched training samples.
        batch_targets:
            A torch.Tensor of shape (batch_count, response_count) containing
            the responses corresponding to each batched training sample.
        batch_nn_targets:
            A torch.Tensor of shape (batch_count, nn_count, response_count)
            containing the responses corresponding to the nearest neighbors
            of each batched training sample.


        kwargs:
            Addition parameters to be passed to the kernel, possibly including
            additional hyperparameter dicts and a metric keyword.
    """

    def __init__(
        self,
        kernel_eps,
        nu,
        length_scale,
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()

        self.length_scale = nn.Parameter(torch.tensor(length_scale))
        self.eps = kernel_eps
        self.nu = nu
        self.batch_indices = batch_indices
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets
        self.variance_mode = "diagonal"
        self.apply_sigma_sq = True

    def forward(self, x):
        """
        Produce the output of a MuyGPs custom PyTorch layer.

        Args:
            x: A torch.Tensor of shape `(batch_count, feature_count)`
            containing feature vector to be regressed by the MuyGPs_layer.

        Returns:
            A torch.Tensor of shape `(batch_count, response_count)` listing the
            predicted response for each of the batch elements.
        """

        crosswise_dists = crosswise_distances(
            x,
            x,
            self.batch_indices,
            self.batch_nn_indices,
            metric="l2",
        )

        pairwise_dists = pairwise_distances(
            x, self.batch_nn_indices, metric="l2"
        )

        Kcross = kernel_func(
            crosswise_dists,
            nu=self.nu,
            length_scale=self.length_scale,
        )
        K = kernel_func(
            pairwise_dists,
            nu=self.nu,
            length_scale=self.length_scale,
        )
        predictions = _muygps_compute_solve(
            K, Kcross, self.batch_nn_targets, self.eps
        )

        sigma_sq = _analytic_sigma_sq_optim(K, self.batch_nn_targets, self.eps)

        if self.variance_mode is None:
            return predictions
        elif self.variance_mode == "diagonal":
            variances = _muygps_compute_diagonal_variance(K, Kcross, self.eps)
            if self.apply_sigma_sq is True:
                if len(sigma_sq) == 1:
                    variances *= sigma_sq
                else:
                    variances = torch.outer(variances, sigma_sq)
        else:
            raise NotImplementedError(
                f"Variance mode {self.variance_mode} is not implemented."
            )

        return predictions, variances, sigma_sq


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
            A torch.Tensor of shape `(test_count, feature_count)' containing
            the test features to be regressed.
        train_features:
            A torch.Tensor of shape `(train_count, feature_count)' containing
            the training features.
        train_responses:
            A torch.Tensor of shape `(train_count, response_count)' containing
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


class MultivariateMuyGPs_layer(nn.Module):
    """
    MuyGPs model written as a custom PyTorch layer using nn.Module.

    Implements the MuyGPs algorithm as articulated in [muyskens2021muygps]_. See
    documentation on MuyGPs class for more detail.

    PyTorch does not currently support the Bessel function required to compute
    the Matern kernel for non-special values of :math:`\\nu`, e.g. 1/2, 3/2,
    5/2, and :math:`\\infty`. The MuyGPs layer allows the lengthscale parameter
    :math:`\\rho` to be trained (provided an initial value by the user) as well
    as the homoscedastic :math:`\\varepsilon` noise parameter.

    The MuyGPs layer returns the posterior mean, posterior variance, and a
    vector of :math:`\\sigma^2` indicating the scale parameter associated
    with the posterior variance of each dimension of the response.

    :math:`\\sigma^2` is the only parameter assumed to be a training target by
    default, and is treated differently from all other hyperparameters. All
    other training targets must be manually specified in the construction of
    a MuyGPs_layer object.

    Example:
        >>> from MuyGPyS.pytorch.muygps_layer import MuyGPs_layer
        >>> num_models = 10
        >>> kernel_eps = 1e-3 * torch.ones(10,)
        >>> nu = 1/2 * torch.ones(10,)
        >>> length_scale = 1.0 * torch.ones(10,)
        >>> batch_indices = torch.arange(100,)
        >>> batch_nn_indices = torch.arange(100,)
        >>> batch_targets = torch.ones(100,)
        >>> batch_nn_targets = torch.ones(100,)
        >>> muygps_layer_object = MuyGPs_layer(kernel_eps, nu, length_scale,
        batch_indices, batch_nn_indices, batch_targets, batch_nn_targets)



    Args:
        kernel_eps:
            A torch.Tensor of shape num_models containing the hyperparameter
            corresponding to the aleatoric uncertainty in the
            data for each model.
        nu:
            A torch.Tensor of shape num_models containing the smoothness
            parameter in the Matern kernel for each model. Allowed to take on
            values 1/2, 3/2, 5/2, or torch.inf.
        length_scale:
            A torch.Tensor of shape num_models containing the length scale
            parameter in the Matern kernel for each model.
        batch_indices:
            A torch.Tensor of shape (batch_count,) containing the indices of
            the training data to be sampled for training.
        batch_nn_indices:
            A torch.Tensor of shape (batch_count, nn_count) containing the
            indices of the k nearest neighbors of the batched training samples.
        batch_targets:
            A torch.Tensor of shape (batch_count, response_count) containing
            the responses corresponding to each batched training sample.
        batch_nn_targets:
            A torch.Tensor of shape (batch_count, nn_count, response_count)
            containing the responses corresponding to the nearest neighbors
            of each batched training sample.


        kwargs:
            Addition parameters to be passed to the kernel, possibly including
            additional hyperparameter dicts and a metric keyword.
    """

    def __init__(
        self,
        num_models,
        kernel_eps,
        nu_vals,
        length_scales,
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()
        self.num_models = num_models
        self.length_scale = nn.Parameter(torch.Tensor(length_scales))
        self.eps = kernel_eps
        self.nu = nu_vals
        self.batch_indices = batch_indices
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets

    def forward(self, x):
        """
        Produce the output of a MuyGPs custom PyTorch layer.

        Args:
            x: A torch.Tensor of shape `(batch_count, feature_count)`
            containing feature vector to be regressed by the MuyGPs_layer.

        Returns:
            A torch.Tensor of shape `(batch_count, response_count)` listing the
            predicted response for each of the batch elements.
        """
        crosswise_dists = crosswise_distances(
            x,
            x,
            self.batch_indices,
            self.batch_nn_indices,
            metric="l2",
        )

        pairwise_dists = pairwise_distances(
            x, self.batch_nn_indices, metric="l2"
        )

        batch_count, nn_count, response_count = self.batch_nn_targets.shape

        Kcross = torch.zeros(batch_count, nn_count, response_count)
        K = torch.zeros(batch_count, nn_count, nn_count, response_count)

        for i in range(self.num_models):
            Kcross[:, :, i] = kernel_func(
                crosswise_dists,
                nu=self.nu[i],
                length_scale=self.length_scale[i],
            )

            K[:, :, :, i] = kernel_func(
                pairwise_dists,
                nu=self.nu[i],
                length_scale=self.length_scale[i],
            )

        predictions = torch.zeros(batch_count, response_count)
        variances = torch.zeros(batch_count, response_count)
        sigma_sq = torch.zeros(
            response_count,
        )

        for i in range(self.num_models):
            predictions[:, i] = _muygps_compute_solve(
                K[:, :, :, i],
                Kcross[:, :, i],
                self.batch_nn_targets[:, :, i].reshape(
                    batch_count, nn_count, 1
                ),
                self.eps,
            ).reshape(batch_count)
            variances[:, i] = _muygps_compute_diagonal_variance(
                K[:, :, :, i], Kcross[:, :, i], self.eps
            )
            sigma_sq[i] = _analytic_sigma_sq_optim(
                K[:, :, :, i],
                self.batch_nn_targets[:, :, i].reshape(
                    batch_count, nn_count, 1
                ),
                self.eps,
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
    MultivariateMuyGPs_layer in its structure.

    Args:
        model:
            A custom PyTorch.nn.Module object containing at least one
            MuyGPs_layer.
        test_features:
            A torch.Tensor of shape `(test_count, feature_count)' containing
            the test features to be regressed.
        train_features:
            A torch.Tensor of shape `(train_count, feature_count)' containing
            the training features.
        train_responses:
            A torch.Tensor of shape `(train_count, response_count)' containing
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


def kernel_func(
    dist_matrix: torch.Tensor, nu: float, length_scale: float
) -> torch.Tensor:
    if nu == 1 / 2:
        return matern_05_fn(dist_matrix, length_scale=length_scale)

    if nu == 3 / 2:
        return matern_15_fn(dist_matrix, length_scale=length_scale)

    if nu == 5 / 2:
        return matern_25_fn(dist_matrix, length_scale=length_scale)

    if nu == torch.inf:
        return matern_inf_fn(dist_matrix, length_scale=length_scale)
    else:
        return matern_gen_fn(dist_matrix, nu, length_scale=length_scale)
