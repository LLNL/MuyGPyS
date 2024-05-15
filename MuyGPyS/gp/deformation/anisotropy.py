# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math as mm
from MuyGPyS._src.mpi_utils import mpi_chunk
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation.deformation_fn import DeformationFn
from MuyGPyS.gp.deformation.metric import MetricFn
from MuyGPyS.gp.hyperparameter import ScalarParam, VectorParam, NamedVectorParam
from MuyGPyS.gp.hyperparameter.experimental import (
    HierarchicalParam,
    NamedHierarchicalVectorParam,
)


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
        name = "length_scale"
        params = length_scale._params
        # This is brittle and should be refactored
        if all(isinstance(p, ScalarParam) for p in params):
            self.length_scale = NamedVectorParam(name, length_scale)
        elif all(isinstance(p, HierarchicalParam) for p in params):
            self.length_scale = NamedHierarchicalVectorParam(name, length_scale)
        else:
            raise ValueError(
                "Expected uniform vector of ScalarParam or HierarchicalParam type for length_scale"
            )
        self.metric = metric

    def __call__(self, dists: mm.ndarray, **length_scales) -> mm.ndarray:
        """
        Apply anisotropic deformation to an elementwise difference tensor.

        This function is not intended to be invoked directly by a user. It is
        instead functionally incorporated into some
        :class:`MuyGPyS.gp.kernels.KernelFn` in its constructor.

        Args:
            dists:
                A tensor of pairwise differences of shape
                `(..., feature_count)` representing the difference in feature
                dimensions between sets of observables.
            batch_features:
                A `(batch_count, feature_count)` matrix of features to be used
                with a hierarchical hyperparameter. `None` otherwise.
            length_scale:
                A floating point length scale, or a vector of `(knot_count,)`
                knot length scales.
        Returns:
            A crosswise distance matrix of shape `(data_count, nn_count)` or a
            pairwise distance tensor of shape
            `(data_count, nn_count, nn_count)` whose last two dimensions are
            pairwise distance matrices.
        """
        if dists.shape[-1] != len(self.length_scale):
            raise ValueError(
                f"Difference tensor of shape {dists.shape} must have final "
                f"dimension size of {len(self.length_scale)}"
            )
        length_scale = self.length_scale(**length_scales)
        # This is brittle and similar to what we do in Isotropy.
        if isinstance(length_scale, mm.ndarray) and len(length_scale.shape) > 0:
            shape = [None] * dists.ndim
            shape[0] = slice(None)
            shape[-1] = slice(None)
            length_scale = length_scale.T[tuple(shape)]
        return self.metric(dists / length_scale)

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
