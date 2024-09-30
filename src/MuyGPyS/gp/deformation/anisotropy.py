# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math as mm
from MuyGPyS._src.mpi_utils import mpi_chunk
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation.deformation_fn import DeformationFn
from MuyGPyS.gp.deformation.metric import MetricFn
from MuyGPyS.gp.hyperparameter import VectorParam, NamedVectorParam


@auto_str
class Anisotropy(DeformationFn):
    """
    An anisotropic deformation model.

    Anisotropy parameterizes a scaled elementwise distance function
    :math:`d_\\ell(\\cdot, \\cdot)`, and is paramterized by a vector-valued
    :math:`\\mathbf{\\ell}>0` length scale hyperparameter.

    .. math::
         d_\\ell(\\mathbf{x}, \\mathbf{y}) =
         \\sum_{i=0}^d \\frac{d(\\mathbf{x}_i, \\mathbf{y}_i)}{\\ell_i}

    Args:
        metric:
            A MetricFn object defining the behavior of the feature metric space.
        length_scales:
            Keyword arguments `length_scale#`, mapping to scalar
            hyperparameters.
    """

    def __init__(
        self,
        metric: MetricFn,
        length_scale: VectorParam,
    ):
        self.metric = metric
        self.length_scale = NamedVectorParam("length_scale", length_scale)

    def __call__(self, dists: mm.ndarray, **length_scales) -> mm.ndarray:
        """
        Apply anisotropic deformation to an elementwise difference tensor.

        This function is not intended to be invoked directly by a user. It is
        instead functionally incorporated into some
        :class:`MuyGPyS.gp.kernels.KernelFn` in its constructor.

        Args:
            dists:
                A difference tensor of shape `(..., feature_count)` listing
                feature-wise differences among a set of observables.
            batch_features:
                A `(batch_count, feature_count)` matrix of features to be used
                with a hierarchical hyperparameter. `None` otherwise.
            length_scale:
                A floating point length scale, or a vector of `(knot_count,)`
                knot length scales.
        Returns:
            A scaled distance tensor of the same shape as :math:`dists` less
            the final dimension.
        """
        if dists.shape[-1] != len(self.length_scale):
            raise ValueError(
                f"Difference tensor of shape {dists.shape} must have final "
                f"dimension size of {len(self.length_scale)}"
            )
        return self.metric(dists / self.length_scale(**length_scales))

    @mpi_chunk(return_count=1)
    def pairwise_tensor(
        self,
        data: mm.ndarray,
        nn_indices: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        """
        Compute a pairwise difference tensor among sets of nearest neighbors.

        Takes a full dataset of records of interest `data` and produces the
        pairwise differences for each feature dimension between the elements
        indicated by each row of `nn_indices`.

        Args:
            data:
                The data matrix of shape `(batch_count, feature_count)`
                containing batch elements.
            nn_indices:
                An integral matrix of shape (batch_count, nn_count) listing the
                nearest neighbor indices for the batch of data points.

        Returns:
            A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
            containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
            nearest neighbor difference tensors corresponding to each of the
            batch elements.
        """
        return self.metric.pairwise_differences(data, nn_indices)

    @mpi_chunk(return_count=1)
    def crosswise_tensor(
        self,
        data: mm.ndarray,
        nn_data: mm.ndarray,
        data_indices: mm.ndarray,
        nn_indices: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        """
        Compute a crosswise difference tensor between data and their nearest
        neighbors.

        Takes full datasets of records of interest `data` and neighbor
        candidates `nn_data` and produces a difference vector between each
        element of `data` indicated by `data_indices` and each of the nearest
        neighbors in `nn_data` as indicated by the corresponding rows of
        `nn_indices`. `data` and `nn_data` can refer to the same dataset.

        Args:
            data:
                The data matrix of shape `(data_count, feature_count)`
                containing batch elements.
            nn_data:
                The data matrix of shape `(candidate_count, feature_count)`
                containing the universe of candidate neighbors for the batch
                elements. Might be the same as `data`.
            indices:
                An integral vector of shape `(batch_count,)` containing the
                indices of the batch.
            nn_indices:
                An integral matrix of shape (batch_count, nn_count) listing the
                nearest neighbor indices for the batch of data points.

        Returns:
            A tensor of shape `(batch_count, nn_count, feature_count)` whose
            last two dimensions indicate difference vectors between the feature
            dimensions of each batch element and those of its nearest neighbors.
        """
        return self.metric.crosswise_differences(
            data, nn_data, data_indices, nn_indices
        )
