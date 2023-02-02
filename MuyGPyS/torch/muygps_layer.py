# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs PyTorch implementation
"""

import torch
from torch import nn
import numpy as np
from MuyGPyS._src.gp.distance.torch import (
    _pairwise_distances,
    _crosswise_distances,
)


from MuyGPyS._src.gp.muygps.torch import (
    _muygps_compute_solve,
    _muygps_compute_diagonal_variance,
)
from MuyGPyS._src.optimize.sigma_sq.torch import _analytic_sigma_sq_optim

from MuyGPyS._src.gp.noise.torch import _homoscedastic_perturb

from MuyGPyS._src.gp.kernels.torch import (
    _matern_05_fn,
    _matern_15_fn,
    _matern_25_fn,
    _matern_inf_fn,
    _matern_gen_fn,
)


class MuyGPs_layer(nn.Module):
    """
    MuyGPs model written as a custom PyTorch layer using nn.Module.

    Implements the MuyGPs algorithm as articulated in [muyskens2021muygps]_. See
    documentation on MuyGPs class for more detail.

    The MuyGPs_layer class only supports the Matern kernel currently. More
    kernels will be added to the torch module of MuyGPs in future releases.

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
        >>> from MuyGPyS.torch.muygps_layer import MuyGPs_layer
        >>> kernel_eps = 1e-3
        >>> nu = 1/2
        >>> length_scale = 1.0
        >>> batch_indices = torch.arange(100,)
        >>> batch_nn_indices = torch.arange(100,)
        >>> batch_targets = torch.ones(100,)
        >>> batch_nn_targets = torch.ones(100,)
        >>> muygps_layer_object = MuyGPs_layer(
        ... kernel_eps,
        ... nu,
        ... length_scale,
        ... batch_indices,
        ... batch_nn_indices,
        ... batch_targets,
        ... batch_nn_targets)



    Args:
        kernel_eps:
            A hyperparameter corresponding to the aleatoric uncertainty in the
            data.
        nu:
            A smoothness parameter allowed to take on values 1/2, 3/2, 5/2, or
            :math:`\\infty`.
        length_scale:
            The length scale parameter in the Matern kernel.
        batch_indices:
            A torch.Tensor of shape `(batch_count,)` containing the indices of
            the training data to be sampled for training.
        batch_nn_indices:
            A torch.Tensor of shape `(batch_count, nn_count)` containing the
            indices of the k nearest neighbors of the batched training samples.
        batch_targets:
            A torch.Tensor of shape `(batch_count, response_count)` containing
            the responses corresponding to each batched training sample.
        batch_nn_targets:
            A torch.Tensor of shape `(batch_count, nn_count, response_count)`
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

        Returns:
            A torch.Tensor of shape `(batch_count, response_count)` listing the
            predicted response for each of the batch elements.
        """

        crosswise_dists = _crosswise_distances(
            x,
            x,
            self.batch_indices,
            self.batch_nn_indices,
            metric="l2",
        )

        pairwise_dists = _pairwise_distances(
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
            _homoscedastic_perturb(K, self.eps), Kcross, self.batch_nn_targets
        )

        sigma_sq = _analytic_sigma_sq_optim(
            _homoscedastic_perturb(K, self.eps), self.batch_nn_targets
        )

        if self.variance_mode is None:
            return predictions
        elif self.variance_mode == "diagonal":
            variances = _muygps_compute_diagonal_variance(
                _homoscedastic_perturb(K, self.eps), Kcross
            )
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


class MultivariateMuyGPs_layer(nn.Module):
    """
    Multivariate MuyGPs model written as a custom PyTorch layer using nn.Module.

    Implements the MuyGPs algorithm as articulated in [muyskens2021muygps]_. See
    documentation on MuyGPs class for more detail.

    The MultivariateMuyGPs_layer class only supports the Matern kernel
    currently. More kernels will be added to the torch module of MuyGPs in
    future releases.

    PyTorch does not currently support the Bessel function required to compute
    the Matern kernel for non-special values of :math:`\\nu`, e.g. 1/2, 3/2,
    5/2, and :math:`\\infty`. The MuyGPs layer allows the lengthscale parameter
    :math:`\\rho` to be trained (provided an initial value by the user) as well
    as the homoskedastic :math:`\\varepsilon` noise parameter.

    The MuyGPs layer returns the posterior mean, posterior variance, and a
    vector of :math:`\\sigma^2` indicating the scale parameter associated
    with the posterior variance of each dimension of the response.

    :math:`\\sigma^2` is the only parameter assumed to be a training target by
    default, and is treated differently from all other hyperparameters. All
    other training targets must be manually specified in the construction of
    a MuyGPs_layer object.

    Example:
        >>> from MuyGPyS.torch.muygps_layer import MultivariateMuyGPs_layer
        >>> num_models = 10
        >>> kernel_eps = 1e-3 * torch.ones(10,)
        >>> nu = 1/2 * torch.ones(10,)
        >>> length_scale = 1.0 * torch.ones(10,)
        >>> batch_indices = torch.arange(100,)
        >>> batch_nn_indices = torch.arange(100,)
        >>> batch_targets = torch.ones(100,)
        >>> batch_nn_targets = torch.ones(100,)
        >>> muygps_layer_object = MultivariateMuyGPs_layer(
        ... num_models,
        ... kernel_eps,
        ... nu,
        ... length_scale,
        ... batch_indices,
        ... batch_nn_indices,
        ... batch_targets,
        ... batch_nn_targets)



    Args:
        num_models:
            The number of MuyGPs models to be used in the layer.
        kernel_eps:
            A torch.Tensor of shape `(num_models,)` containing the hyperparameter
            corresponding to the aleatoric uncertainty in the
            data for each model.
        nu:
            A torch.Tensor of shape `(num_models,)` containing the smoothness
            parameter in the Matern kernel for each model. Allowed to take on
            values 1/2, 3/2, 5/2, or :math:`\\infty`.
        length_scale:
            A torch.Tensor of shape `(num_models,)` containing the length scale
            parameter in the Matern kernel for each model.
        batch_indices:
            A torch.Tensor of shape `(batch_count,)` containing the indices of
            the training data to be sampled for training.
        batch_nn_indices:
            A torch.Tensor of shape `(batch_count, nn_count)` containing the
            indices of the k nearest neighbors of the batched training samples.
        batch_targets:
            A torch.Tensor of shape `(batch_count, response_count)` containing
            the responses corresponding to each batched training sample.
        batch_nn_targets:
            A torch.Tensor of shape `(batch_count, nn_count, response_count)`
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
        # self.length_scale = length_scales
        self.eps = kernel_eps
        self.nu = nu_vals
        self.batch_indices = batch_indices
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets

    def forward(self, x):
        """
        Produce the output of a MuyGPs custom PyTorch layer.

        Returns:
            A torch.Tensor of shape `(batch_count, response_count)` listing the
            predicted response for each of the batch elements.
        """
        crosswise_dists = _crosswise_distances(
            x,
            x,
            self.batch_indices,
            self.batch_nn_indices,
            metric="l2",
        )

        pairwise_dists = _pairwise_distances(
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
                _homoscedastic_perturb(K[:, :, :, i], self.eps[i]),
                Kcross[:, :, i],
                self.batch_nn_targets[:, :, i].reshape(
                    batch_count, nn_count, 1
                ),
            ).reshape(batch_count)
            variances[:, i] = _muygps_compute_diagonal_variance(
                _homoscedastic_perturb(K[:, :, :, i], self.eps[i]),
                Kcross[:, :, i],
            )
            sigma_sq[i] = _analytic_sigma_sq_optim(
                _homoscedastic_perturb(K[:, :, :, i], self.eps[i]),
                self.batch_nn_targets[:, :, i].reshape(
                    batch_count, nn_count, 1
                ),
            )
        return predictions, variances, sigma_sq


def kernel_func(
    dist_tensor: torch.Tensor, nu: float, length_scale: float
) -> torch.Tensor:
    """
    Generate kernel tensors using the Matern kernel given an input distance
    tensor. Currently only supports the Matern kernel, but more kernels will
    be added in future releases.

    Args:
        dist_matrix:
            A torch.Tensor distance tensor on which to evaluate the kernel.
        nu:
            The smoothness hyperparameter in the Matern kernel.
        length_scale:
            The lengthscale hyperparameter in the Matern kernel.

    Returns:
        A torch.Tensor containing the kernel matrix evaluated for the given
        input values.
    """
    if nu == 1 / 2:
        return _matern_05_fn(dist_tensor, length_scale=length_scale)

    if nu == 3 / 2:
        return _matern_15_fn(dist_tensor, length_scale=length_scale)

    if nu == 5 / 2:
        return _matern_25_fn(dist_tensor, length_scale=length_scale)

    if nu == torch.inf:
        return _matern_inf_fn(dist_tensor, length_scale=length_scale)
    else:
        return _matern_gen_fn(dist_tensor, nu, length_scale=length_scale)
