# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


# TO DO: Build MultivariateMuyGPs_layer
# TO DO: Add thorough documentation
# TO DO:


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
        self.batch_inds = batch_indices
        self.batch_nn_inds = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets
        self.variance_mode = "diagonal"
        self.apply_sigma_sq = True

    def forward(self, x):

        crosswise_dists = crosswise_distances(
            x,
            x,
            self.batch_inds,
            self.batch_nn_inds,
            metric="l2",
        )

        pairwise_dists = pairwise_distances(x, self.batch_nn_inds, metric="l2")

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
    test_x,
    train_x,
    train_responses,
    nbrs_lookup,
    nn_count,
    variance_mode="diagonal",
    apply_sigma_sq=True,
):

    train_x_numpy = model.embedding(train_x).detach().numpy()
    test_x_numpy = model.embedding(test_x).detach().numpy()

    nn_indices_train, _ = nbrs_lookup._get_nns(train_x_numpy, nn_count=nn_count)
    nn_indices_test, _ = nbrs_lookup._get_nns(test_x_numpy, nn_count=nn_count)

    nn_indices_test = torch.from_numpy(nn_indices_test.astype(np.int64))

    train_x_numpy = torch.from_numpy(train_x_numpy).float()
    test_x_numpy = torch.from_numpy(test_x_numpy).float()

    train_nn_targets = torch.from_numpy(
        train_responses[nn_indices_train, :]
    ).float()
    test_nn_targets = torch.from_numpy(
        train_responses[nn_indices_test, :]
    ).float()

    test_count = test_x_numpy.shape[0]

    crosswise_dists = crosswise_distances(
        test_x_numpy,
        train_x_numpy,
        np.arange(test_count),
        nn_indices_test,
        metric="l2",
    )

    pairwise_dists = pairwise_distances(
        train_x_numpy, nn_indices_test, metric="l2"
    )

    Kcross = kernel_func(
        crosswise_dists,
        nu=model.GP_layer.nu,
        length_scale=model.GP_layer.length_scale,
    )
    K = kernel_func(
        pairwise_dists,
        nu=model.GP_layer.nu,
        length_scale=model.GP_layer.length_scale,
    )

    predictions = _muygps_compute_solve(
        K, Kcross, test_nn_targets, model.GP_layer.eps
    )

    sigma_sq = _analytic_sigma_sq_optim(K, test_nn_targets, model.GP_layer.eps)

    if variance_mode is None:
        return predictions
    elif variance_mode == "diagonal":
        variances = _muygps_compute_diagonal_variance(
            K, Kcross, model.GP_layer.eps
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


class MultivariateMuyGPs_layer(nn.Module):
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
        self.batch_inds = batch_indices
        self.batch_nn_inds = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets

    def forward(self, x):

        crosswise_dists = crosswise_distances(
            x,
            x,
            self.batch_inds,
            self.batch_nn_inds,
            metric="l2",
        )

        pairwise_dists = pairwise_distances(x, self.batch_nn_inds, metric="l2")

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
    test_x,
    train_x,
    train_responses,
    nbrs_lookup,
    nn_count,
):

    train_x_numpy = model.embedding(train_x).detach().numpy()
    test_x_numpy = model.embedding(test_x).detach().numpy()

    nn_indices_train, _ = nbrs_lookup._get_nns(train_x_numpy, nn_count=nn_count)
    nn_indices_test, _ = nbrs_lookup._get_nns(test_x_numpy, nn_count=nn_count)

    nn_indices_test = torch.from_numpy(nn_indices_test.astype(np.int64))

    train_x_numpy = torch.from_numpy(train_x_numpy).float()
    test_x_numpy = torch.from_numpy(test_x_numpy).float()

    train_nn_targets = torch.from_numpy(
        train_responses[nn_indices_train, :]
    ).float()
    test_nn_targets = torch.from_numpy(
        train_responses[nn_indices_test, :]
    ).float()

    test_count = test_x_numpy.shape[0]

    crosswise_dists = crosswise_distances(
        test_x_numpy,
        train_x_numpy,
        np.arange(test_count),
        nn_indices_test,
        metric="l2",
    )

    pairwise_dists = pairwise_distances(
        train_x_numpy, nn_indices_test, metric="l2"
    )

    (
        batch_count,
        nn_count,
        response_count,
    ) = model.GP_layer.batch_nn_targets.shape

    Kcross = torch.zeros(test_count, nn_count, response_count)
    K = torch.zeros(test_count, nn_count, nn_count, response_count)

    for i in range(num_responses):
        Kcross[:, :, i] = kernel_func(
            crosswise_dists,
            nu=model.GP_layer.nu[i],
            length_scale=model.GP_layer.length_scale[i],
        )

        K[:, :, :, i] = kernel_func(
            pairwise_dists,
            nu=model.GP_layer.nu[i],
            length_scale=model.GP_layer.length_scale[i],
        )

    batch_count, nn_count, response_count = test_nn_targets.shape

    predictions = torch.zeros(batch_count, response_count)
    variances = torch.zeros(batch_count, response_count)
    sigma_sq = torch.zeros(
        response_count,
    )

    for i in range(model.GP_layer.num_models):
        predictions[:, i] = _muygps_compute_solve(
            K[:, :, :, i],
            Kcross[:, :, i],
            test_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
            model.GP_layer.eps,
        ).reshape(batch_count)
        variances[:, i] = _muygps_compute_diagonal_variance(
            K[:, :, :, i], Kcross[:, :, i], model.GP_layer.eps
        )
        sigma_sq[i] = _analytic_sigma_sq_optim(
            K[:, :, :, i],
            test_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
            model.GP_layer.eps,
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
