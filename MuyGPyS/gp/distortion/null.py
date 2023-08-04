# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Dict, Callable, Union

from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.hyperparameter.experimental import (
    HierarchicalNonstationaryHyperparameter,
)


class NullDistortion:
    def __init__(
        self,
        metric: Callable,
        length_scale: Union[
            ScalarHyperparameter, HierarchicalNonstationaryHyperparameter
        ],
    ):
        self.length_scale = length_scale
        self._dist_fn = metric

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("NullDistortion cannot be called!")

    def populate_length_scale(
        self, hyperparameters: Dict, *args, **kwargs
    ) -> None:
        """
        A no-op placeholder function for NullDistortion objects.
        """

    def get_opt_fn(self, *args, **kwargs):
        raise NotImplementedError(
            "NullDistortion cannot be used in optimization!"
        )

    def get_opt_params(self, *args, **kwargs):
        raise NotImplementedError(
            "NullDistortion cannot be used in optimization!"
        )
