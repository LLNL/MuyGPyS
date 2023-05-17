# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from typing import Callable, Dict, List, Tuple, Union

import MuyGPyS._src.math as mm
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.hyperparameter.experimental import (
    HierarchicalNonstationaryHyperparameter,
)


@auto_str
class IsotropicDistortion:
    def __init__(
        self,
        metric: Callable,
        length_scale: Union[
            ScalarHyperparameter, HierarchicalNonstationaryHyperparameter
        ],
    ):
        self.length_scale = length_scale
        self._dist_fn = metric

    def __call__(
        self, diffs: mm.ndarray, length_scale: Union[float, mm.ndarray]
    ) -> mm.ndarray:
        length_scale_array = self._get_length_scale_array(
            mm.array, diffs.shape, length_scale
        )
        return self._dist_fn(diffs / length_scale_array)

    @staticmethod
    def _get_length_scale_array(
        array_fn: Callable,
        target_shape: mm.ndarray,
        length_scale: Union[float, mm.ndarray],
    ) -> mm.ndarray:
        # make sure length_scale is broadcastable when its shape is (batch_count,)
        shape = (-1,) + (1,) * (len(target_shape) - 1)
        return mm.reshape(array_fn(length_scale), shape)

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
        self.length_scale.append_lists("length_scale", names, params, bounds)
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
        opt_fn = self.length_scale.apply(fn, "length_scale")
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
