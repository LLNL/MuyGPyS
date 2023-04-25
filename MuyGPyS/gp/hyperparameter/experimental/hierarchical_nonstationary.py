# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

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
