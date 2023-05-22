# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable, Union, List, Tuple

from scipy.stats.qmc import LatinHypercube

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import (
    _pairwise_differences,
    _crosswise_differences,
)
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter


class HierarchicalNonstationaryHyperparameter:
    """
    A MuyGPs kernel or model hierarchical nonstationary hyperparameter.

    HierarchicalNonstationaryHyperparameter are defined by a set of knots,
    initial guesses for the hyperparameter values at each of those knots,
    and a higher-level GP kernel.

    Args:
        knot_features:
            Tensor of floats of shape `(knot_count, feature_count)`
            containing the feature vectors for each knot.
        knot_values:
            List of scalar hyperparameters of length `knot_count`
            containing the initial values and optimization bounds for each knot.
            Float values will be converted to fixed scalar hyperparameters.
        kernel:
            Initialized higher-level GP kernel.
    """

    def __init__(
        self,
        knot_features: mm.ndarray,
        knot_values: Union[mm.ndarray, List[ScalarHyperparameter]],
        kernel,
    ):
        """
        Initialize a hierarchical nonstationary hyperparameter.
        """
        if len(knot_values) != len(knot_features):
            raise ValueError(
                "knot_features and knot_values must have the same length"
            )
        self._knot_features = knot_features
        self._knot_value_params = [
            knot_value
            if isinstance(knot_value, ScalarHyperparameter)
            else ScalarHyperparameter(knot_value)
            for knot_value in knot_values
        ]
        self._kernel = kernel
        lower_K = self._kernel(_pairwise_differences(self._knot_features))
        self._solve = mm.linalg.solve(lower_K, self._knot_values())

    def _knot_values(self) -> mm.ndarray:
        return mm.ndarray(param() for param in self._knot_value_params)

    def __call__(self, batch_features) -> mm.ndarray:
        """
        Value accessor.

        Returns:
            The current value of the hierarchical nonstationary hyperparameter.
        """
        lower_Kcross = self._kernel(
            _crosswise_differences(batch_features, self._knot_features)
        )
        return lower_Kcross @ self._solve

    def fixed(self) -> bool:
        """
        Report whether the parameter is fixed, and is to be ignored during optimization.

        Returns:
            `True` if fixed, `False` otherwise.
        """
        return all(param.fixed() for param in self._knot_value_params)

    def get_bounds(self) -> Tuple[float, float]:
        """
        Bounds accessor.

        Returns:
            The lower and upper bound tuple.
        """
        raise NotImplementedError(
            "HierarchicalNonstationaryHyperparameter does not support optimization bounds directly. "
            "Set bounds on individual knot values instead."
        )

    @staticmethod
    def _get_knot_key(name: str, index: int) -> str:
        return f"{name}_knot{index}"

    def apply(self, fn: Callable, name: str) -> Callable:
        if any(param.fixed() for param in self._knot_value_params):

            def applied_fn(*args, **kwargs):
                for index, param in enumerate(self._knot_value_params):
                    if param.fixed():
                        knot_name = self._get_knot_key(name, index)
                        kwargs.setdefault(knot_name, param())
                return fn(*args, **kwargs)

            return applied_fn

        return fn

    def append_lists(
        self,
        name: str,
        names: List[str],
        params: List[float],
        bounds: List[Tuple[float, float]],
    ):
        for index, param in enumerate(self._knot_value_params):
            knot_name = self._get_knot_key(name, index)
            param.append_lists(knot_name, names, params, bounds)


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
