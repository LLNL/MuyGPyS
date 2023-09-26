# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from typing import Callable, Dict, List, Tuple

import MuyGPyS._src.math as mm


class DeformationFn:
    """
    The base deformation functor class.

    Contains some function :math:`d_\\ell(\\cdot, \\cdot)` that computes scalar
    similarities of pairs of points, possibly applying a non-Euclidean
    deformation to the feature space.
    """

    def __init__(
        self,
        metric: Callable,
        length_scale,
    ):
        self.length_scale = length_scale
        self._dist_fn = metric

    def __call__(self, diffs: mm.ndarray, **kwargs) -> mm.ndarray:
        raise NotImplementedError("Cannot call DeformationFn base class!")

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
            "Cannot call DeformationFn base class functions!"
        )

    def populate_length_scale(self, hyperparameters: Dict) -> None:
        """
        Populates the hyperparameter dictionary of a KernelFn object with any
        parameters of the DeformationFn object.

        Args:
            hyperparameters:
                A dict containing the hyperparameters of a KernelFn object.
        """
        raise NotImplementedError(
            "Cannot call DeformationFn base class functions!"
        )

    def embed_fn(self, fn: Callable) -> Callable:
        """
        Augments a function to automatically apply the deformation to a
        difference tensor.

        Args:
            fn:
                A Callable with signature
                `(diffs, *args, **kwargs) -> mm.ndarray` taking a difference
                tensor `diffs`.

        Returns:
            A new Callable that applies the deformation to `diffs`, possibly
            changing its tensor dimensionality.
        """
        raise NotImplementedError(
            "Cannot call DeformationFn base class functions!"
        )
