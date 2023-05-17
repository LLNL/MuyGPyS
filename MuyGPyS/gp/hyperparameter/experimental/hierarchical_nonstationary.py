# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable

from scipy.stats.qmc import LatinHypercube

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import (
    _pairwise_differences,
    _crosswise_differences,
)


class HierarchicalNonstationaryHyperparameter:
    """
    A MuyGPs kernel or model hierarchical nonstationary hyperparameter.

    HierarchicalNonstationaryHyperparameter are defined by a set of knots,
    initial guesses for the hyperparameter values at each of those knots,
    and a lower-level GP kernel.

    Args:
        knot_features:
            Tensor of floats of shape `(knot_count, feature_count)`
            containing the feature vectors for each knot.
        knot_values:
            Tensor of floats of shape `(knot_count, 1)`
            containing the initial values at each knot.
        kernel:
            Initialized lower-level GP kernel.
    """

    def __init__(self, knot_features, knot_values, kernel):
        self.kernel = kernel
        self.knot_features = knot_features
        lower_K = self.kernel(_pairwise_differences(self.knot_features))
        self.solve = mm.linalg.solve(lower_K, knot_values)

    def __call__(self, batch_features) -> mm.ndarray:
        lower_Kcross = self.kernel(
            _crosswise_differences(batch_features, self.knot_features)
        )
        return lower_Kcross @ self.solve

    def apply(self, fn: Callable, name: str) -> Callable:
        raise NotImplementedError(
            "optimizing hierarchical parameters is not yet supported"
        )


def sample_knots(feature_count: int, knot_count: int) -> mm.ndarray:
    """
    Samples knots from feature matrix.

    Args:
        feature_count:
            Dimension of feature vectors.
        knot_count:
            Number of knots to sample.

    Returns:
        Tensor of floats of shape `(knot_count, feature_count)`
        containing the sampled feature vectors for each knot.
    """
    return LatinHypercube(feature_count, centered=True).random(knot_count)
