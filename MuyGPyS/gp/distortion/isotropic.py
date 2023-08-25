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
    """
    An isotropic distance model.

    IsotropicDistortion parameterizes a scaled elementwise distance function
    :math:`d(\\cdot, \\cdot)`, and is paramterized by a scalar :math:`\\ell>0`
    length scale hyperparameter.

    .. math::
         d_\\ell(\\mathbf{x}, \\mathbf{y}) =
         \\sum_{i=0}^d \\frac{d(\\mathbf{x}_i, \\mathbf{y}_i)}{\\ell}

    Args:
        metric:
            A callable metric function that takes a tensor of shape
            `(..., feature_count)` whose last dimension lists the elementwise
            differences between a pair of feature vectors and returns a tensor
            of shape `(...)`, having collapsed the last dimension into a
            scalar difference.
        length_scale:
            Some scalar nonnegative hyperparameter object.
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
        """
        Apply isotropic distortion to an elementwise difference tensor.

        This function is not intended to be invoked directly by a user. It is
        instead functionally incorporated into some
        :class:`MuyGPyS.gp.kernels.KernelFn` in its constructor.

        Args:
            diffs:
                A tensor of pairwise differences of shape
                `(..., feature_count)`.
            length_scale:
                A floating point length scale, or a vector of `(knot_count,)`
                knot length scales.
        Returns:
            A crosswise distance matrix of shape `(data_count, nn_count)` or a
            pairwise distance tensor of shape
            `(data_count, nn_count, nn_count)` whose last two dimensions are
            pairwise distance matrices.
        """
        length_scale_array = self._get_length_scale_array(
            diffs.shape, length_scale
        )
        return self._dist_fn(diffs / length_scale_array)

    @staticmethod
    def _get_length_scale_array(
        target_shape: mm.ndarray,
        length_scale: Union[float, mm.ndarray],
    ) -> mm.ndarray:
        # make sure length_scale is broadcastable when its shape is (batch_count,)
        # NOTE[MWP] there is probably a better way to do this
        shape = (-1,) + (1,) * (len(target_shape) - 1)
        return mm.reshape(mm.promote(length_scale), shape)

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

    def populate_length_scale(self, hyperparameters: Dict) -> None:
        """
        Populates the hyperparameter dictionary of a KernelFn object with
        `self.length_scale` of the IsotropicDistortion object.

        Args:
            hyperparameters:
                A dict containing the hyperparameters of a KernelFn object.
        """
        hyperparameters["length_scale"] = self.length_scale
