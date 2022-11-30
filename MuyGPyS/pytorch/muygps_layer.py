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

    def forward(self, x):

        c_d = crosswise_distances(
            x,
            x,
            self.batch_inds,
            self.batch_nn_inds,
            metric="l2",
        )

        p_d = pairwise_distances(x, self.batch_nn_inds, metric="l2")

        rho = self.length_scale

        if self.nu == 1 / 2:
            Kcross = torch.exp(-c_d / self.length_scale)
            K = torch.exp(-p_d / self.length_scale)

        if self.nu == 3 / 2:
            Kcross = (
                1 + torch.sqrt(torch.tensor(3)) * c_d / self.length_scale
            ) * torch.exp(
                -torch.sqrt(torch.tensor(3)) * c_d / self.length_scale
            )
            K = (
                1 + torch.sqrt(torch.tensor(3)) * p_d / self.length_scale
            ) * torch.exp(
                -torch.sqrt(torch.tensor(3)) * p_d / self.length_scale
            )

        if self.nu == 5 / 2:
            Kcross = (
                1
                + torch.sqrt(torch.tensor(5)) * c_d / self.length_scale
                + 5.0 * c_d**2 / 3.0 / self.length_scale**2
            ) * torch.exp(
                -torch.sqrt(torch.tensor(5)) * c_d / self.length_scale
            )
            K = (
                1
                + torch.sqrt(torch.tensor(5)) * p_d / self.length_scale
                + 5.0 * p_d**2 / 3.0 / self.length_scale**2
            ) * torch.exp(
                -torch.sqrt(torch.tensor(5)) * p_d / self.length_scale
            )

        if self.nu == torch.inf:
            Kcross = torch.exp(-(c_d**2) / 2 / (self.length_scale**2))
            K = torch.exp(-(p_d**2) / 2 / (self.length_scale**2))

        batch_count, nn_count, response_count = self.batch_nn_targets.shape

        predictions = torch.matmul(
            Kcross.reshape(batch_count, 1, nn_count),
            torch.linalg.solve(
                K + self.eps * torch.eye(nn_count), self.batch_nn_targets
            ),
        )

        variances = 1 - torch.sum(
            Kcross
            * torch.linalg.solve(
                K + self.eps * torch.eye(nn_count),
                Kcross.reshape(batch_count, nn_count, 1),
            ).reshape(batch_count, nn_count),
            axis=1,
        )

        sigma_sq = torch.sum(
            torch.einsum(
                "ijk,ijk->ik",
                self.batch_nn_targets,
                torch.linalg.solve(
                    K + self.eps * torch.eye(nn_count), self.batch_nn_targets
                ),
            ),
            axis=0,
        ) / (nn_count * batch_count)

        return predictions, variances, sigma_sq


def predict(model, test_x, train_x, train_responses, nbrs_lookup, nn_count):

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

    ###compute GP prediction here
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

    batch_count, nn_count, response_count = test_nn_targets.shape

    predictions = torch.matmul(
        Kcross.reshape(batch_count, 1, nn_count),
        torch.linalg.solve(
            K + model.GP_layer.eps * torch.eye(nn_count), test_nn_targets
        ),
    )

    variances = 1 - torch.sum(
        Kcross
        * torch.linalg.solve(
            K + model.GP_layer.eps * torch.eye(nn_count),
            Kcross.reshape(batch_count, nn_count, 1),
        ).reshape(batch_count, nn_count),
        axis=1,
    )

    sigma_sq = torch.sum(
        torch.einsum(
            "ijk,ijk->ik",
            test_nn_targets,
            torch.linalg.solve(
                K + model.GP_layer.eps * torch.eye(nn_count), test_nn_targets
            ),
        ),
        axis=0,
    ) / (nn_count * batch_count)

    return predictions, variances, sigma_sq


def predict_fixed_nns(
    model, test_x, train_x, train_responses, nbrs_lookup, nn_count
):

    train_x_numpy = train_x.detach().numpy()
    test_x_numpy = test_x.detach().numpy()

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

    ###compute GP prediction here
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

    batch_count, nn_count, response_count = test_nn_targets.shape

    predictions = torch.matmul(
        Kcross.reshape(batch_count, 1, nn_count),
        torch.linalg.solve(
            K + model.GP_layer.eps * torch.eye(nn_count), test_nn_targets
        ),
    )

    variances = 1 - torch.sum(
        Kcross
        * torch.linalg.solve(
            K + model.GP_layer.eps * torch.eye(nn_count),
            Kcross.reshape(batch_count, nn_count, 1),
        ).reshape(batch_count, nn_count),
        axis=1,
    )

    sigma_sq = torch.sum(
        torch.einsum(
            "ijk,ijk->ik",
            test_nn_targets,
            torch.linalg.solve(
                K + model.GP_layer.eps * torch.eye(nn_count), test_nn_targets
            ),
        ),
        axis=0,
    ) / (nn_count * batch_count)

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
