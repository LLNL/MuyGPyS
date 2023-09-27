# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs PyTorch implementation
"""
import MuyGPyS._src.math.torch as torch
from MuyGPyS import config
from MuyGPyS._src.math.torch import nn
from MuyGPyS.gp.tensors import (
    pairwise_tensor,
    crosswise_tensor,
)

from MuyGPyS.gp.multivariate_muygps import MultivariateMuyGPS as MMuyGPS

if config.state.backend != "torch":
    import warnings

    warnings.warn(
        f"torch-only code cannot be run in {config.state.backend} mode"
    )


class MultivariateMuyGPs_layer(nn.Module):
    """
    Multivariate MuyGPs model written as a custom PyTorch layer using nn.Module.

    Implements the MuyGPs algorithm as articulated in [muyskens2021muygps]_. See
    documentation on MuyGPs class for more detail.

    The MultivariateMuyGPs_layer class only supports the RBF and Matern kernels
    currently. More kernels will be added to the torch module of MuyGPs in
    future releases.

    PyTorch does not currently support the Bessel function required to compute
    the Matern kernel for non-special case smoothness values of :math:`\\nu`,
    e.g. 1/2, 3/2, 5/2, and :math:`\\infty`. The MuyGPs layer allows the
    lengthscale parameter :math:`\\rho` to be trained (provided an initial value
    by the user) as well as the homoscedastic :math:`\\tau^2` noise prior
    variance parameter.

    The MultivariateMuyGPs layer returns the posterior mean, posterior variance
    , and a vector of :math:`\\sigma^2` indicating the scale parameter
    associated with the posterior variance of each dimension of the response.

    Example:
        >>> from MuyGPyS.torch.muygps_layer import MultivariateMuyGPs_layer
        >>> multivariate_muygps_model = MMuyGPs(
        ...     Matern(
        ...         smoothness=ScalarParam(0.5),
        ...         deformation=Isotropy(
        ...             metric=l2,
        ...             length_scale=ScalarParam(1.0)
        ...         ),
        ...     ),
        ...     noise=HomoscedasticNoise(1e-5),
        ... )
        >>> batch_indices = torch.arange(100,)
        >>> batch_nn_indices = torch.arange(100,)
        >>> batch_targets = torch.ones(100,)
        >>> batch_nn_targets = torch.ones(100,)
        >>> muygps_layer_object = MultivariateMuyGPs_layer(
        ... multivariate_muygps_model,
        ... batch_indices,
        ... batch_nn_indices,
        ... batch_targets,
        ... batch_nn_targets)



    Args:
        multivariate_muygps_model:
            A MultivariateMuyGPS object to be used in the final GP layer.
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
        multivariate_muygps_model: MMuyGPS,
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()
        self.multivariate_muygps_model = multivariate_muygps_model
        self.batch_indices = batch_indices
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets

    def forward(self, x):
        """
        Produce the output of a MuyGPs custom PyTorch layer.

        Returns
        -------
        predictions:
            A torch.ndarray of shape `(batch_count, response_count)` whose rows
            are the predicted response for each of the given batch feature.
        variances:
            A torch.ndarray of shape `(batch_count, response_count)`
            consisting of the diagonal elements of the posterior variance.
        """
        crosswise_diffs = crosswise_tensor(
            x,
            x,
            self.batch_indices,
            self.batch_nn_indices,
        )

        pairwise_diffs = pairwise_tensor(x, self.batch_nn_indices)

        batch_count, nn_count, response_count = self.batch_nn_targets.shape

        Kcross = torch.zeros(batch_count, nn_count, response_count)
        K = torch.zeros(batch_count, nn_count, nn_count, response_count)

        for i, model in enumerate(self.multivariate_muygps_model.models):
            Kcross[:, :, i] = model.kernel(crosswise_diffs)
            K[:, :, :, i] = model.kernel(pairwise_diffs)

        variances = self.multivariate_muygps_model.posterior_variance(K, Kcross)

        predictions = self.multivariate_muygps_model.posterior_mean(
            K, Kcross, self.batch_nn_targets
        )

        return predictions, variances
