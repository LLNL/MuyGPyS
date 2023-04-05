# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import _F2, _l2

from typing import List, Tuple, Callable

from MuyGPyS.gp.kernels import (
    append_optim_params_lists,
    apply_hyperparameter,
    Hyperparameter,
)


class IsotropicDistortion:
    def __init__(self, metric: str, length_scale: Hyperparameter):
        self.metric = metric
        self.length_scale = length_scale
        if metric == "l2":
            self._dist_fn = _l2
        elif metric == "F2":
            self._dist_fn = _F2
        else:
            raise ValueError(f"Metric {metric} is not supported!")

    def __call__(self, diffs: mm.ndarray) -> mm.ndarray:
        return self._dist_fn(diffs / self.length_scale())

    def get_optim_params(
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
        append_optim_params_lists(
            self.length_scale, "length_scale", names, params, bounds
        )
        return names, params, bounds

    def get_opt_fn(self) -> Callable:
        """
        Return a kernel function with fixed parameters set.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` and assumes
        that optimization parameters will be passed as keyword arguments.

        Returns:
            A function implementing the kernel where all fixed parameters are
            set. The function expects keyword arguments corresponding to current
            hyperparameter values for unfixed parameters.
        """
        return self._get_opt_fn(self._dist_fn, self.length_scale)

    @staticmethod
    def _get_opt_fn(
        dist_fn: Callable, length_scale: Hyperparameter
    ) -> Callable:
        opt_fn = apply_hyperparameter(dist_fn, length_scale, "length_scale")
        return opt_fn
