# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, List, Tuple

from scipy.stats.qmc import LatinHypercube

import MuyGPyS._src.math as mm
from MuyGPyS.gp.hyperparameter import VectorParam, NamedVectorParam
from MuyGPyS.gp.noise import HomoscedasticNoise


class HierarchicalParameter:
    """
    A MuyGPs kernel or model hierarchical nonstationary hyperparameter.

    HierarchicalNonstationaryHyperparameter are defined by a set of knots,
    initial guesses for the hyperparameter values at each of those knots,
    and a higher-level GP kernel.

    Args:
        knot_features:
            Tensor of floats of shape `(knot_count, feature_count)`
            containing the feature vectors for each knot.
        knot_params:
            List of scalar hyperparameters of length `knot_count`
            containing the initial values and optimization bounds for each knot.
        kernel:
            Initialized higher-level GP kernel.
    """

    def __init__(
        self,
        knot_features: mm.ndarray,
        knot_params: VectorParam,
        kernel,
        noise: HomoscedasticNoise = HomoscedasticNoise(1e-5),
    ):
        """
        Initialize a hierarchical nonstationary hyperparameter.
        """
        self._knot_count = len(knot_params)
        if self._knot_count != len(knot_features):
            raise ValueError(
                "knot_features and knot_values must have the same length"
            )
        self._knot_features = knot_features
        self._knot_params = knot_params
        self._kernel = kernel
        self._Kin_higher = self._kernel(
            self._kernel.deformation.pairwise_tensor(
                self._knot_features, mm.arange(knot_features.shape[0])
            )
        )
        self._noise = noise

    def __call__(self, batch_features, **kwargs) -> mm.ndarray:
        """
        Value accessor.

        Returns:
            The current value of the hierarchical nonstationary hyperparameter.
        """
        raise NotImplementedError(
            "__call__ not implemented for base HierarchicalParameter."
        )

    def fixed(self) -> bool:
        """
        Report whether the parameter is fixed, and is to be ignored during
        optimization.

        Returns:
            `True` if fixed, `False` otherwise.
        """
        return self._knot_params.fixed()

    def get_bounds(self) -> Tuple[float, float]:
        """
        Bounds accessor.

        Returns:
            The lower and upper bound tuple.
        """
        raise NotImplementedError(
            "HierarchicalNonstationaryHyperparameter does not support "
            "optimization bounds directly. Set bounds on individual knot "
            "values instead."
        )


class NamedHierarchicalParameter(HierarchicalParameter):
    def __init__(self, name: str, rhs: HierarchicalParameter):
        self._knot_count = rhs._knot_count
        self._knot_features = rhs._knot_features
        self._params = NamedVectorParam(name, rhs._knot_params)
        self._Kin_higher = rhs._Kin_higher
        self._kernel = rhs._kernel
        self._noise = rhs._noise
        self._name = name

    def name(self) -> str:
        return self._name

    def knot_values(self) -> mm.ndarray:
        return self._params()

    def __call__(self, batch_features, **kwargs) -> float:
        params, kwargs = self._params.filter_kwargs(**kwargs)
        solve = mm.linalg.solve(
            self._Kin_higher + self._noise() * mm.eye(self._knot_count),
            self._params(**params),
        )
        lower_Kcross = self._kernel(
            self._kernel.deformation.crosswise_tensor(
                batch_features,
                self._knot_features,
                mm.arange(batch_features.shape[0]),
                mm.arange(self._knot_features.shape[0]),
            )
        )
        return mm.squeeze(lower_Kcross @ solve)

    def filter_kwargs(self, **kwargs) -> Tuple[Dict, Dict]:
        params, kwargs = self._params.filter_kwargs(**kwargs)
        lower = dict()
        lower[self._name] = self(kwargs["batch_features"], **params)
        return lower, kwargs

    def apply_fn(self, fn: Callable, name: str) -> Callable:
        def applied_fn(*args, **kwargs):
            lower, kwargs = self.filter_kwargs(**kwargs)
            return fn(*args, **lower, **kwargs)

        return applied_fn

    def apply_embedding_fn(
        self, fn: Callable, deformation_fn: Callable
    ) -> Callable:

        def embedded_fn(dists, *args, **kwargs):
            lower, kwargs = self.filter_kwargs(**kwargs)
            return fn(deformation_fn(dists, **lower), *args, **kwargs)

        return embedded_fn

    def append_lists(
        self,
        names: List[str],
        params: List[float],
        bounds: List[Tuple[float, float]],
    ):
        return self._params.append_lists(names, params, bounds)

    def populate(self, hyperparameters: Dict) -> None:
        self._params.populate(hyperparameters)


class NamedHierarchicalVectorParameter(NamedVectorParam):
    def __init__(self, name: str, param: VectorParam):
        self._params = [
            NamedHierarchicalParameter(name + str(i), p)
            for i, p in enumerate(param._params)
        ]
        self._name = name

    def filter_kwargs(self, **kwargs) -> Tuple[Dict, Dict]:
        params = {
            key: kwargs[key] for key in kwargs if key.startswith(self._name)
        }
        kwargs = {
            key: kwargs[key] for key in kwargs if not key.startswith(self._name)
        }
        if "batch_features" in kwargs:
            for p in self._params:
                params.setdefault(
                    p.name(), p(kwargs["batch_features"], **params)
                )
        return params, kwargs


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
    return LatinHypercube(feature_count, scramble=False).random(knot_count)
