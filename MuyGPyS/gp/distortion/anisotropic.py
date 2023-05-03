# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple, Callable, Dict

import MuyGPyS._src.math as mm
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.hyperparameter import (
    append_scalar_optim_params_list,
    apply_scalar_hyperparameter,
    ScalarHyperparameter,
)


@auto_str
class AnisotropicDistortion:
    def __init__(self, metric: Callable, **length_scales):
        self._dist_fn = metric
        self.length_scale = length_scales
        for i, key in enumerate(self.length_scale.keys()):
            if key != "length_scale" + str(i) or not isinstance(
                self.length_scale[key], ScalarHyperparameter
            ):
                raise ValueError(
                    "Anisotropic model expects either one keyword argument for"
                    "each feature in the dataset labeled length_scalei for the"
                    "ith feature with indexing beginning at zero, with each"
                    "corresponding value being a ScalarHyperparameter."
                )

    def __call__(self, diffs: mm.ndarray, **length_scales) -> mm.ndarray:
        length_scale_array = self._get_length_scale_array(
            mm.array, diffs.shape[-1], **length_scales
        )
        return self._dist_fn(diffs / length_scale_array)

    @staticmethod
    def _get_length_scale_array(
        array_fn: Callable, target_length: float, **length_scales
    ) -> mm.ndarray:
        AnisotropicDistortion._lengths_agree(len(length_scales), target_length)
        if callable(length_scales["length_scale0"]) is True:
            length_scales = {ls: length_scales[ls]() for ls in length_scales}
        return array_fn([value for value in length_scales.values()])

    @staticmethod
    def _lengths_agree(found_length, target_length):
        if target_length != found_length and found_length != 1:
            raise ValueError(
                f"Number of lengthscale parameters ({found_length}) "
                f"must match number of features ({target_length}) or be 1 "
                "(Isotropic model)."
            )

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
            append_scalar_optim_params_list(
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
            fn = apply_scalar_hyperparameter(
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
