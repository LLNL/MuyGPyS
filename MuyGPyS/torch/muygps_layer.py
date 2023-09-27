# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs PyTorch implementation
"""
from MuyGPyS import config
from MuyGPyS._src.math.torch import nn
from MuyGPyS.gp.deformation import Isotropy
from MuyGPyS.gp.hyperparameter import ScalarParam
from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS.gp.tensors import (
    pairwise_tensor,
    crosswise_tensor,
)


if config.state.backend != "torch":
    import warnings

    warnings.warn(
        f"torch-only code cannot be run in {config.state.backend} mode"
    )


class MuyGPs_layer(nn.Module):
    """
    MuyGPs model written as a custom PyTorch layer using nn.Module.

    Implements the MuyGPs algorithm as articulated in [muyskens2021muygps]_. See
    documentation on MuyGPs class for more detail.

    The MuyGPs_layer class only supports the Matern kernel currently. More
    kernels will be added to the torch module of MuyGPs in future releases.

    PyTorch does not currently support the Bessel function required to compute
    the Matern kernel for non-special case smoothness values of :math:`\\nu`,
    e.g. 1/2, 3/2, 5/2, and :math:`\\infty`. The MuyGPs layer allows the
    lengthscale parameter :math:`\\rho` to be trained (provided an initial value
    by the user) as well as the homoscedastic :math:`\\tau^2` noise prior
    variance.

    The MuyGPs layer returns the posterior mean, posterior variance, and a
    vector of :math:`\\sigma^2` indicating the scale parameter associated
    with the posterior variance of each dimension of the response.

    Example:
        >>> from MuyGPyS.torch.muygps_layer import MuyGPs_layer
        >>> muygps_model = MuyGPS(
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
        >>> muygps_layer_object = MuyGPs_layer(
        ... muygps_model,
        ... batch_indices,
        ... batch_nn_indices,
        ... batch_targets,
        ... batch_nn_targets)



    Args:
        muygps_model:
            A MuyGPs object providing the Gaussian Process final layer.
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
        muygps_model: MuyGPS,
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()
        if not isinstance(muygps_model.kernel.deformation, Isotropy):
            raise NotImplementedError(
                "MuyGPyS/torch optimization does not support "
                f"{type(muygps_model.kernel.deformation)} deformations"
            )
        if not isinstance(
            muygps_model.kernel.deformation.length_scale, ScalarParam
        ):
            raise NotImplementedError(
                "MuyGPyS/torch optimization does not support "
                f"{type(muygps_model.kernel.deformation.length_scale)} "
                "length scales"
            )
        self.muygps_model = muygps_model
        self.length_scale = muygps_model.kernel.deformation.length_scale._val
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
            A torch.ndarray of shape `(batch_count,response_count)`
            consisting of the diagonal elements of the posterior variance.
        """
        self.muygps_model._make()

        crosswise_diffs = crosswise_tensor(
            x,
            x,
            self.batch_indices,
            self.batch_nn_indices,
        )

        pairwise_diffs = pairwise_tensor(x, self.batch_nn_indices)

        Kcross = self.muygps_model.kernel(crosswise_diffs)
        K = self.muygps_model.kernel(pairwise_diffs)

        predictions = self.muygps_model.posterior_mean(
            K, Kcross, self.batch_nn_targets
        )

        variances = self.muygps_model.posterior_variance(K, Kcross)

        return predictions, variances
