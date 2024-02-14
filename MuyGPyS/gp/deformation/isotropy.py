# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from typing import Callable, Dict, List, Optional, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.mpi_utils import mpi_chunk
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation.deformation_fn import DeformationFn
from MuyGPyS.gp.deformation.metric import MetricFn
from MuyGPyS.gp.hyperparameter import ScalarParam


@auto_str
class Isotropy(DeformationFn):
    """
    An isotropic deformation model.

    Isotropy defines a scaled elementwise distance function
    :math:`d_ell(\\cdot, \\cdot)`, and is paramterized by a scalar
    :math:`\\ell>0` length scale hyperparameter.

    .. math::
         d_\\ell(\\mathbf{x}, \\mathbf{y}) =
         \\sum_{i=0}^d \\frac{d(\\mathbf{x}_i, \\mathbf{y}_i)}{\\ell}

    Args:
        metric:
            A MetricFn object defining the behavior of the feature metric space.
        length_scale:
            Some scalar nonnegative hyperparameter object.
    """

    def __init__(
        self,
        metric: MetricFn,
        length_scale: ScalarParam,
    ):
        if not isinstance(length_scale, ScalarParam):
            raise ValueError(
                "Expected ScalarParam type for length_scale, not "
                f"{type(length_scale)}"
            )
        self.length_scale = length_scale
        self.metric = metric

    def __call__(
        self, dists: mm.ndarray, length_scale: Optional[float] = None, **kwargs
    ) -> mm.ndarray:
        """
        Apply isotropic deformation to an elementwise difference tensor.

        This function is not intended to be invoked directly by a user. It is
        instead functionally incorporated into some
        :class:`MuyGPyS.gp.kernels.KernelFn` in its constructor.

        Args:
            dists:
                A tensor of distances between sets of observables.
            length_scale:
                A floating point length scale.
        Returns:
            A scaled distance matrix of the same shape as shape `(data_count, nn_count)` or a
            pairwise distance tensor of shape
            `(data_count, nn_count, nn_count)` whose last two dimensions are
            pairwise distance matrices.
        """
        if length_scale is None:
            length_scale = self.length_scale()
        return self.metric.apply_length_scale(dists, length_scale)

    def get_opt_params(
        self,
    ) -> Tuple[List[str], List[float], List[Tuple[float, float]]]:
        """
        Report lists of unfixed hyperparameter names, values, and bounds.

        Returns
        -------
            names:
                A list of unfixed hyperparameter names.
            params:
                A list of unfixed hyperparameter values.
            bounds:
                A list of unfixed hyperparameter bound tuples.
        """
        names: List[str] = []
        params: List[float] = []
        bounds: List[Tuple[float, float]] = []
        self.length_scale.append_lists("length_scale", names, params, bounds)
        return names, params, bounds

    def populate_length_scale(self, hyperparameters: Dict) -> None:
        """
        Populates the hyperparameter dictionary of a KernelFn object with
        `self.length_scale` of the Isotropy object.

        Args:
            hyperparameters:
                A dict containing the hyperparameters of a KernelFn object.
        """
        hyperparameters["length_scale"] = self.length_scale

    def embed_fn(self, fn: Callable) -> Callable:
        """
        Augments a function to automatically apply the deformation to a
        difference tensor.

        Args:
            fn:
                A Callable with signature
                `(diffs, *args, **kwargs) -> mm.ndarray` taking a difference
                tensor `diffs` with shape `(..., feature_count)`.

        Returns:
            A new Callable that applies the deformation to `diffs`, removing
            the last tensor dimension by collapsing the feature-wise differences
            into scalar distances. Also adds a `length_scale` kwarg, making the
            function drivable by keyword optimization.
        """

        def embedded_fn(dists, *args, length_scale=None, **kwargs):
            return fn(self(dists, length_scale=length_scale), *args, **kwargs)

        return embedded_fn

    @mpi_chunk(return_count=1)
    def pairwise_tensor(
        self,
        data: mm.ndarray,
        nn_indices: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        """
        Compute a pairwise distance tensor among sets of nearest neighbors.

        Takes a full dataset of records of interest `data` and produces the
        pairwise distances between the elements indicated by each row of
        `nn_indices`.

        Args:
            data:
                The data matrix of shape `(batch_count, feature_count)`
                containing batch elements.
            nn_indices:
                An integral matrix of shape (batch_count, nn_count) listing the
                nearest neighbor indices for the batch of data points.

        Returns:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing the
            `(nn_count, nn_count)`-shaped pairwise nearest neighbor distance
            tensors corresponding to each of the batch elements.
        """
        return self.metric.pairwise_distances(data, nn_indices)

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
        Compute a crosswise distance tensor between data and their nearest
        neighbors.

        Takes full datasets of records of interest `data` and neighbor
        candidates `nn_data` and produces a scalar distance between each
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
            A tensor of shape `(batch_count, nn_count)` whose second dimension
            indicates distance vectors between each batch element and its
            nearest neighbors.
        """
        return self.metric.crosswise_distances(
            data, nn_data, data_indices, nn_indices
        )


@auto_str
class DifferenceIsotropy(Isotropy):
    """
    An isotropic deformation model that reasons about differences rather than
    distances.

    Isotropy defines a scaled elementwise distance function
    :math:`d_ell(\\cdot, \\cdot)`, and is paramterized by a scalar
    :math:`\\ell>0` length scale hyperparameter.

    .. math::
         d_\\ell(\\mathbf{x}, \\mathbf{y}) =
         \\sum_{i=0}^d \\frac{d(\\mathbf{x}_i, \\mathbf{y}_i)}{\\ell}

    Args:
        metric:
            A MetricFn object defining the behavior of the feature metric space.
        length_scale:
            Some scalar nonnegative hyperparameter object.
    """

    def __call__(
        self, dists: mm.ndarray, length_scale: Optional[float] = None, **kwargs
    ) -> mm.ndarray:
        """
        Apply isotropic deformation to an elementwise difference tensor.

        This function is not intended to be invoked directly by a user. It is
        instead functionally incorporated into some
        :class:`MuyGPyS.gp.kernels.KernelFn` in its constructor.

        Args:
            dists:
                A tensor of distances between sets of observables.
            length_scale:
                A floating point length scale.
        Returns:
            A scaled distance matrix of the same shape as shape `(data_count, nn_count)` or a
            pairwise distance tensor of shape
            `(data_count, nn_count, nn_count)` whose last two dimensions are
            pairwise distance matrices.
        """
        if length_scale is None:
            length_scale = self.length_scale()
        return self.metric(dists / length_scale)

    @mpi_chunk(return_count=1)
    def pairwise_tensor(
        self,
        data: mm.ndarray,
        nn_indices: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        """
        Compute a pairwise distance tensor among sets of nearest neighbors.

        Takes a full dataset of records of interest `data` and produces the
        pairwise distances between the elements indicated by each row of
        `nn_indices`.

        Args:
            data:
                The data matrix of shape `(batch_count, feature_count)`
                containing batch elements.
            nn_indices:
                An integral matrix of shape (batch_count, nn_count) listing the
                nearest neighbor indices for the batch of data points.

        Returns:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing the
            `(nn_count, nn_count)`-shaped pairwise nearest neighbor distance
            tensors corresponding to each of the batch elements.
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
        Compute a crosswise distance tensor between data and their nearest
        neighbors.

        Takes full datasets of records of interest `data` and neighbor
        candidates `nn_data` and produces a scalar distance between each
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
            A tensor of shape `(batch_count, nn_count)` whose second dimension
            indicates distance vectors between each batch element and its
            nearest neighbors.
        """
        return self.metric.crosswise_differences(
            data, nn_data, data_indices, nn_indices
        )
