# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from typing import List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS.gp.deformation.metric import MetricFn


class DeformationFn:
    """
    The base deformation functor class.

    Contains some function :math:`d_\\ell(\\cdot, \\cdot)` that computes scalar
    similarities of pairs of points, possibly applying a non-Euclidean
    deformation to the feature space.
    """

    def __init__(
        self,
        metric: MetricFn,
        length_scale,
    ):
        raise NotImplementedError("Cannot initialize DeformationFn base class!")

    def __call__(self, dists: mm.ndarray, **kwargs) -> mm.ndarray:
        raise NotImplementedError("Cannot call DeformationFn base class!")

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
        raise NotImplementedError(
            "Cannot call DeformationFn base class functions!"
        )

    def pairwise_tensor(
        self,
        data: mm.ndarray,
        nn_indices: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        """
        Compute a pairwise metric tensor among sets of nearest neighbors.

        Takes a full dataset of records of interest `data` and produces the
        pairwise metric needed by the deformation between the elements indicated
        by each row of `nn_indices`.

        Args:
            data:
                The data matrix of shape `(batch_count, feature_count)`
                containing batch elements.
            nn_indices:
                An integral matrix of shape (batch_count, nn_count) listing the
                nearest neighbor indices for the batch of data points.

        Returns:
            A tensor of shape `(batch_count, nn_count, nn_count, ...)`
            containing the `(nn_count, nn_count, ...)`-shaped pairwise nearest
            neighbor metric tensors corresponding to each of the batch elements.
        """
        raise NotImplementedError(
            "Cannot call DeformationFn base class functions!"
        )

    def crosswise_tensor(
        self,
        data: mm.ndarray,
        nn_data: mm.ndarray,
        data_indices: mm.ndarray,
        nn_indices: mm.ndarray,
        **kwargs,
    ) -> mm.ndarray:
        """
        Compute a crosswise metric tensor between data and their nearest
        neighbors.

        Takes full datasets of records of interest `data` and neighbor
        candidates `nn_data` and produces a metric comparison between each
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
            A tensor of shape `(batch_count, nn_count, ...)` whose later
            dimensions list a metric between each feature of each batch element
            and its nearest neighbors.
        """
        raise NotImplementedError(
            "Cannot call DeformationFn base class functions!"
        )
