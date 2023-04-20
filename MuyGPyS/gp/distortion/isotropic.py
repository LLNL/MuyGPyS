# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from typing import Callable, Dict, List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import _F2, _l2
from MuyGPyS.gp.hyperparameter import (
    append_scalar_optim_params_list,
    apply_scalar_hyperparameter,
)


class IsotropicDistortion:
    def __init__(self, metric: str, **length_scale):
        self.metric = metric
        self.length_scale = length_scale["length_scale"]
        if metric == "l2":
            self._dist_fn = _l2
        elif metric == "F2":
            self._dist_fn = _F2
        else:
            raise ValueError(f"Metric {metric} is not supported!")

    def __call__(self, diffs: mm.ndarray, **length_scale) -> mm.ndarray:
        length_scale = length_scale["length_scale"]
        return self._dist_fn(diffs / length_scale)

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
        append_scalar_optim_params_list(
            self.length_scale, "length_scale", names, params, bounds
        )
        return names, params, bounds

    def get_opt_fn(self, fn) -> Callable:
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
        opt_fn = apply_scalar_hyperparameter(
            fn, self.length_scale, "length_scale"
        )
        return opt_fn

    def populate_length_scale(self, hyperparameters: Dict) -> Dict:
        """
        Populates the hyperparameter dictionary of a KernelFn object with
        `self.length_scale` of the IsotropicDistortion object.

        Args:
        hyperparameters:
            A dict containing the hyperparameters of a KernelFn object.

        Returns:
            An updated hyperparameter dictionary.
        """
        hyperparameters["length_scale"] = self.length_scale
        return hyperparameters
