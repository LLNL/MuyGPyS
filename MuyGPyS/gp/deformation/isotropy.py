# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from typing import Callable, Dict, List, Optional, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.deformation.deformation_fn import DeformationFn
from MuyGPyS.gp.hyperparameter import ScalarParam


@auto_str
class Isotropy(DeformationFn):
    """
    An isotropic deformation model.

    Isotropy defines a scaled elementwise distance function
    :math:`d_ell(\\cdot, \\cdot)`, and is paramterized by a scalar
    :math:`\\ell>0` length scale hyperparameter.

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
        length_scale: ScalarParam,
    ):
        if not isinstance(length_scale, ScalarParam):
            raise ValueError(
                "Expected ScalarParam type for length_scale, not "
                f"{type(length_scale)}"
            )
        self.length_scale = length_scale
        self._dist_fn = metric

    def __call__(
        self, diffs: mm.ndarray, length_scale: Optional[float] = None, **kwargs
    ) -> mm.ndarray:
        """
        Apply isotropic deformation to an elementwise difference tensor.

        This function is not intended to be invoked directly by a user. It is
        instead functionally incorporated into some
        :class:`MuyGPyS.gp.kernels.KernelFn` in its constructor.

        Args:
            diffs:
                A tensor of pairwise differences of shape
                `(..., feature_count)`.
            length_scale:
                A floating point length scale.
        Returns:
            A crosswise distance matrix of shape `(data_count, nn_count)` or a
            pairwise distance tensor of shape
            `(data_count, nn_count, nn_count)` whose last two dimensions are
            pairwise distance matrices.
        """
        if length_scale is None:
            length_scale = self.length_scale()
        return self._dist_fn(diffs / length_scale)

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

    def populate_length_scale(self, hyperparameters: Dict) -> None:
        """
        Populates the hyperparameter dictionary of a KernelFn object with
        `self.length_scale` of the Isotropy object.

        Args:
            hyperparameters:
                A dict containing the hyperparameters of a KernelFn object.
        """
        hyperparameters["length_scale"] = self.length_scale

    def embed_fn(self, fn: Callable) -> Callable:
        """
        Augments a function to automatically apply the deformation to a
        difference tensor.

        Args:
            fn:
                A Callable with signature
                `(diffs, *args, **kwargs) -> mm.ndarray` taking a difference
                tensor `diffs` with shape `(..., feature_count)`.

        Returns:
            A new Callable that applies the deformation to `diffs`, removing
            the last tensor dimension by collapsing the feature-wise differences
            into scalar distances. Also adds a `length_scale` kwarg, making the
            function drivable by keyword optimization.
        """

        def embedded_fn(diffs, *args, length_scale=None, **kwargs):
            return fn(self(diffs, length_scale=length_scale), *args, **kwargs)

        return embedded_fn
