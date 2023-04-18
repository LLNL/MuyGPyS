# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import _F2, _l2
from MuyGPyS._src.gp.muygps import _get_length_scale_array

from copy import deepcopy
from typing import List, Tuple, Callable, Dict

from MuyGPyS.gp.kernels import (
    append_optim_params_lists,
    apply_hyperparameter,
    Hyperparameter,
)


class AnisotropicDistortion:
    def __init__(self, metric: str, **length_scales):
        self.metric = metric
        self.length_scale = length_scales
        for i, key in enumerate(self.length_scale.keys()):
            if key != "length_scale" + str(i) or not isinstance(
                self.length_scale[key], Hyperparameter
            ):
                raise ValueError(
                    f"Anisotropic model expects either one keyword"
                    f"argument labeled length_scale0 or a keyword argument for"
                    f"each feature in the dataset labeled length_scalei for the"
                    f" ith feature."
                )
        if metric == "l2":
            self._dist_fn = _l2
        elif metric == "F2":
            self._dist_fn = _F2
        else:
            raise ValueError(f"Metric {metric} is not supported!")

    def __call__(self, diffs: mm.ndarray, **length_scales) -> mm.ndarray:
        length_scale_array = _get_length_scale_array(**length_scales)
        if (
            diffs.shape[-1] != len(length_scale_array)
            and len(length_scale_array) != 1
        ):
            raise ValueError(
                f"Number of lengthscale parameters "
                f"({len(length_scale_array)}) must match number of "
                f"features ({diffs.shape[-1]}) or be 1 (Isotropic model)."
            )
        return self._dist_fn(diffs / length_scale_array)

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
        for key in self.length_scale.keys():
            append_optim_params_lists(
                self.length_scale[key],
                key,
                names,
                params,
                bounds,
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
        for key in self.length_scale.keys():
            fn = apply_hyperparameter(
                fn,
                self.length_scale[key],
                key,
            )
        opt_fn = fn
        return opt_fn

    def populate_length_scale(self, hyperparameters: Dict) -> Dict:
        """
        Populates the hyperparameter dictionary of a KernelFn object with
        `self.length_scales` of the AnisotropicDistortion object.

        Args:
        hyperparameters:
            A dict containing the hyperparameters of a KernelFn object.

        Returns:
            An updated hyperparameter dictionary.
        """
        for key in self.length_scale.keys():
            hyperparameters[key] = self.length_scale[key]

        return hyperparameters
