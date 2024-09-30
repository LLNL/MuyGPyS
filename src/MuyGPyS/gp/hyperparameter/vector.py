# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Vector Hyperparameter

Hyperparameter that amounts to a vector of related hyperparameters, such as the
dimension-wise length scales of an anisotropic deformation.
"""

from typing import Callable, Dict, List, Tuple
from MuyGPyS.gp.hyperparameter.scalar import NamedParameter

import MuyGPyS._src.math as mm


class VectorParameter:
    """
    A MuyGPs kernel or model vector-valued Hyperparameter.

    Each element of the VectorParameter is an individual ScalarParameter.

    Args:
        args:
            Individually initialized ScalarParameters.
    """

    def __init__(self, *args):
        """
        Initialize a hyperparameter.
        """
        self._params = list()
        for param in args:
            self._params.append(param)
        self._named = False

    def __len__(self) -> int:
        return len(self._params)

    def set_name(self, name):
        self._name = name
        self._named = True

    def __str__(self, **kwargs):
        ret = f"{type(self).__name__}("
        max_idx = len(self._params) - 1
        for i, param in enumerate(self._params):
            ret += param.__str__()
            if i < max_idx:
                ret += ", "
        ret += ")"
        return ret

    def __call__(self, **kwargs) -> mm.ndarray:
        """
        Value accessor.

        Returns:
            The current value of the hyperparameter.
        """
        return mm.array([param() for param in self._params])

    def fixed(self) -> bool:
        """
        Report whether the parameter is fixed, and is to be ignored during
        optimization.

        Returns:
            `True` if fixed, `False` otherwise.
        """
        return mm.all(param._fixed for param in self._params)


class NamedVectorParameter(VectorParameter):
    def __init__(self, name: str, param: VectorParameter):
        self._params = [
            NamedParameter(name + str(i), p)
            for i, p in enumerate(param._params)
        ]
        self._name = name

    def name(self) -> str:
        return self._name

    def set_defaults(self, **params) -> Dict:
        for p in self._params:
            params.setdefault(p.name(), p())
        return params

    def filter_kwargs(self, **kwargs) -> Tuple[Dict, Dict]:
        params = {
            key: kwargs[key] for key in kwargs if key.startswith(self._name)
        }
        kwargs = {
            key: kwargs[key] for key in kwargs if not key.startswith(self._name)
        }
        params = self.set_defaults(**params)
        return params, kwargs

    def __call__(self, **kwargs) -> mm.ndarray:
        """
        Value accessor.

        Returns:
            The current value of the hyperparameter.
        """
        params, kwargs = self.filter_kwargs(**kwargs)
        return mm.array([param for _, param in params.items()])

    def apply_fn(self, fn: Callable) -> Callable:
        def applied_fn(*args, **kwargs):
            params, kwargs = self.filter_kwargs(**kwargs)
            return fn(*args, **params, **kwargs)

        return applied_fn

    def apply_embedding_fn(
        self, fn: Callable, deformation_fn: Callable
    ) -> Callable:

        def embedded_fn(dists, *args, **kwargs):
            params, kwargs = self.filter_kwargs(**kwargs)
            return fn(deformation_fn(dists, **params), *args, **kwargs)

        return embedded_fn

    def append_lists(
        self,
        names: List[str],
        params: List[float],
        bounds: List[Tuple[float, float]],
    ):
        for p in self._params:
            if not p.fixed():
                names.append(p.name())
                params.append(p())
                bounds.append(p.get_bounds())

    def populate(self, hyperparameters: Dict) -> None:
        for p in self._params:
            p.populate(hyperparameters)
