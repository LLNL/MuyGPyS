# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from typing import Callable, Dict, List, Tuple, Union

import MuyGPyS._src.math as mm
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.hyperparameter.experimental import (
    HierarchicalNonstationaryHyperparameter,
)


class DistortionFn:
    """
    The base distortion functor class.

    Contains some function :math:`d_\\ell(\\cdot, \\cdot)` that computes scalar
    distances of pairs of points.
    """

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
        raise NotImplementedError("Cannot call DistortionFn base class!")

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
            "Cannot call DistortionFn base class functions!"
        )

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
        raise NotImplementedError(
            "Cannot call DistortionFn base class functions!"
        )

    def populate_length_scale(self, hyperparameters: Dict) -> None:
        """
        Populates the hyperparameter dictionary of a KernelFn object with
        `self.length_scale` of the IsotropicDistortion object.

        Args:
            hyperparameters:
                A dict containing the hyperparameters of a KernelFn object.
        """
        raise NotImplementedError(
            "Cannot call DistortionFn base class functions!"
        )
