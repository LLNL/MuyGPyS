# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Metric Function Handling

MuyGPyS includes predefined metric functions with convenience functions for
interacting with the rest of the library.
"""


from typing import Callable

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import _l2, _F2
from MuyGPyS._src.gp.tensors import _crosswise_tensor, _pairwise_tensor


class MetricFn:
    """
    Metric functor class.

    MuyGPyS-compatible metric functions are objects of this class. Creating a
    new metric function is as simple as instantiating a new `MetricFn` object
    with the desired behavior.

    Args:
        differences_metric_fn:
            A Callable taking an ndarray of feature-wise dimensional comparisons
            with shape `(..., feature_count)` that collapses the last dimension
            into scalar distances.
        crosswise_distances_fn:
            A Callable of signature
            `(data, nn_data, data_indices, nn_indices) -> distances` that
            produces a crosswise distance tensor between data and their nearest
            neighbors.
        crosswise_differences_fn:
            A Callable of signature
            `(data, nn_data, data_indices, nn_indices) -> differences` that
            produces a feature dimension-wise crosswise differences tensor
            between data and their nearest neighbors.
        pairwise_distances_fn:
            A Callable of signature `(data, nn_indices) -> distances` that
            produces a pairwise distance tensor among sets of nearest neighbors.
        pairwise_differences_fn:
            A Callable of signature `(data, nn_data) -> differences` that
            produces a feature dimension-wise pairwise differences tensor
            among sets of nearest neighbors.
        apply_length_scale_fn:
            A Callable of signature `(dists) -> dists` that applies a length
            scale parameter appropriately to a distances tensor.
    """

    def __init__(
        self,
        differences_metric_fn: Callable,
        crosswise_differences_fn: Callable,
        pairwise_diffferences_fn: Callable,
        apply_length_scale_fn: Callable,
    ):
        self._differences_metric_fn = differences_metric_fn
        self._crosswise_differences_fn = crosswise_differences_fn
        self._pairwise_differences_fn = pairwise_diffferences_fn
        self._apply_length_scale_fn = apply_length_scale_fn

    def __call__(self, *args, **kwargs):
        return self._differences_metric_fn(*args, **kwargs)

    def crosswise_differences(
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
        return self._crosswise_differences_fn(
            data, nn_data, data_indices, nn_indices
        )

    def crosswise_distances(
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
        return self._differences_metric_fn(
            self._crosswise_differences_fn(
                data, nn_data, data_indices, nn_indices
            )
        )

    def pairwise_differences(
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
        return self._pairwise_differences_fn(data, nn_indices)

    def pairwise_distances(
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
        return self._differences_metric_fn(
            self._pairwise_differences_fn(data, nn_indices)
        )

    def apply_length_scale(
        self, dists: mm.ndarray, length_scale: float
    ) -> mm.ndarray:
        """
        Compute a pairwise distance tensor among sets of nearest neighbors.

        Takes a full dataset of records of interest `data` and produces the
        pairwise distances between the elements indicated by each row of
        `nn_indices`.

        Args:
            dists:
                A distance tensor of any shape.

        Returns:
            A tensor of the same shape that has been element-wise scaled by
            the provided length scale as befits the metric.
        """
        return self._apply_length_scale_fn(dists, length_scale)


l2 = MetricFn(
    differences_metric_fn=_l2,
    crosswise_differences_fn=_crosswise_tensor,
    pairwise_diffferences_fn=_pairwise_tensor,
    apply_length_scale_fn=lambda x, y: x / y,
)
"""
l2 or Euclidean metric function.

Computes the Euclidean distance between points:

.. math::
    d_{\\ell_2}(\\mathbf{x}, \\mathbf{y}) =
        \\left ( \\sum_{i=1}^n (x_i - y_i)^2 \\right )^{1/2}

Args:
    dists:
        A difference tensor of shape `(..., feature_count)`.

Returns:
    A distance tensor of shape `(...)`.
"""

F2 = MetricFn(
    differences_metric_fn=_F2,
    crosswise_differences_fn=_crosswise_tensor,
    pairwise_diffferences_fn=_pairwise_tensor,
    apply_length_scale_fn=lambda x, y: x / y**2,
)
"""
F2 or squared Euclidean metric function.

Computes the Euclidean distance between points:

.. math::
    d_{F_2}(\\mathbf{x}, \\mathbf{y}) =
        \\sum_{i=1}^n (x_i - y_i)^2

Args:
    dists:
        A difference tensor of shape `(..., feature_count)`.

Returns:
    A distance tensor of shape `(...)`.
"""
