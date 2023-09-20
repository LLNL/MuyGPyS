# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple, Callable, Dict

import MuyGPyS._src.math as mm
from MuyGPyS._src.util import auto_str

from MuyGPyS.gp.distortion.distortion_fn import DistortionFn
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter


@auto_str
class AnisotropicDistortion(DistortionFn):
    """
    An isotropic distance model.

    AnisotropicDistortion parameterizes a scaled elementwise distance function
    :math:`d(\\cdot, \\cdot)`, and is paramterized by a vector-valued
    :math:`\\mathbf{\\ell}>0` length scale hyperparameter.

    .. math::
         d_\\ell(\\mathbf{x}, \\mathbf{y}) =
         \\sum_{i=0}^d \\frac{d(\\mathbf{x}_i, \\mathbf{y}_i)}{\\ell_i}

    Args:
        metric:
            A callable metric function that takes a tensor of shape
            `(..., feature_count)` whose last dimension lists the elementwise
            differences between a pair of feature vectors and returns a tensor
            of shape `(...)`, having collapsed the last dimension into a
            scalar difference.
        length_scales:
            Keyword arguments `length_scale#`, mapping to scalar
            hyperparameters.
    """

    def __init__(
        self,
        metric: Callable,
        **length_scales,
    ):
        self._dist_fn = metric
        for i, key in enumerate(length_scales.keys()):
            if key != "length_scale" + str(i):
                raise ValueError(
                    "Anisotropic model expects one keyword argument for each "
                    "feature in the dataset labeled length_scale{i} for the "
                    "ith feature with indexing beginning at zero."
                )
        if not (
            all(
                isinstance(param, ScalarHyperparameter)
                for param in length_scales.values()
            )
        ):
            raise ValueError(
                "Anisotropic model expects all values for the length_scale{i} "
                "keyword arguments to be of type ScalarHyperparameter."
            )
        self.length_scale = length_scales

    def __call__(self, diffs: mm.ndarray, **length_scales) -> mm.ndarray:
        """
        Apply anisotropic distortion to an elementwise difference tensor.

        This function is not intended to be invoked directly by a user. It is
        instead functionally incorporated into some
        :class:`MuyGPyS.gp.kernels.KernelFn` in its constructor.

        Args:
            diffs:
                A tensor of pairwise differences of shape
                `(..., feature_count)`.
            batch_features:
                A `(batch_count, feature_count)` matrix of features to be used
                with a hierarchical hyperparameter. `None` otherwise.
            length_scale:
                A floating point length scale, or a vector of `(knot_count,)`
                knot length scales.
        Returns:
            A crosswise distance matrix of shape `(data_count, nn_count)` or a
            pairwise distance tensor of shape
            `(data_count, nn_count, nn_count)` whose last two dimensions are
            pairwise distance matrices.
        """
        if diffs.shape[-1] != len(self.length_scale):
            raise ValueError(
                f"Difference tensor of shape {diffs.shape} must have final "
                f"dimension size of {len(self.length_scale)}"
            )
        length_scale_array = mm.array(
            [
                length_scales[key]
                if key in length_scales.keys()
                else self.length_scale[key]()
                for key in self.length_scale
            ]
        )
        return self._dist_fn(diffs / length_scale_array)

    # @staticmethod
    # def _get_length_scale_array(
    #     array_fn: Callable,
    #     target_shape: mm.ndarray,
    #     batch_features=None,
    #     **length_scales,
    # ) -> mm.ndarray:
    #     # NOTE[MWP] THIS WILL NOT WORK WITH TORCH OPTIMIZATION.
    #     # We need to eliminate the implicit copy. Will need indirection.
    #     # We should make this whole workflow ifless.
    #     AnisotropicDistortion._lengths_agree(
    #         len(length_scales), target_shape[-1]
    #     )
    #     if callable(length_scales["length_scale0"]) is True:
    #         length_scales = {
    #             ls: length_scales[ls](batch_features) for ls in length_scales
    #         }
    #     # make sure each length_scale array is broadcastable when its shape is (batch_count,)
    #     shape = (1,) * (len(target_shape) - 2) + (-1,)
    #     return array_fn(
    #         [mm.reshape(value, shape) for value in length_scales.values()]
    #     ).T

    # @staticmethod
    # def _lengths_agree(found_length, target_length):
    #     if target_length != found_length and found_length != 1:
    #         raise ValueError(
    #             f"Number of lengthscale parameters ({found_length}) "
    #             f"must match number of features ({target_length}) or be 1 "
    #             "(Isotropic model)."
    #         )

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
        for name, param in self.length_scale.items():
            param.append_lists(name, names, params, bounds)
        return names, params, bounds

    def populate_length_scale(self, hyperparameters: Dict) -> None:
        """
        Populates the hyperparameter dictionary of a KernelFn object with
        `self.length_scales` of the AnisotropicDistortion object.

        Args:
            hyperparameters:
                A dict containing the hyperparameters of a KernelFn object.
        """
        for key, param in self.length_scale.items():
            hyperparameters[key] = param

    def embed_fn(self, fn: Callable) -> Callable:
        """
        Augments a function to automatically apply the distortion to a
        difference tensor.

        Args:
            fn:
                A Callable with signature
                `(diffs, *args, **kwargs) -> mm.ndarray` taking a difference
                tensor `diffs` with shape `(..., feature_count)`.

        Returns:
            A new Callable that applies the distortion to `diffs`, removing
            the last tensor dimension by collapsing the feature-wise differences
            into scalar distances. Propagates any `length_scaleN` kwargs to the
            distortion fn, making the function drivable by keyword optimization.
        """

        def embedded_fn(diffs, *args, length_scale=None, **kwargs):
            length_scales = {
                key: kwargs[key]
                for key in kwargs
                if key.startswith("length_scale")
            }
            kwargs = {
                key: kwargs[key]
                for key in kwargs
                if not key.startswith("length_scale")
            }
            return fn(self(diffs, **length_scales), *args, **kwargs)

        return embedded_fn
