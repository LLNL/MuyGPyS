# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Dict

from MuyGPyS._src.gp.tensors import _F2, _l2
from MuyGPyS.gp.kernels import Hyperparameter


class NullDistortion:
    def __init__(self, metric: str, length_scale: Hyperparameter):
        self.length_scale = length_scale
        self.metric = metric
        if metric == "l2":
            self._dist_fn = _l2
        elif metric == "F2":
            self._dist_fn = _F2
        else:
            raise ValueError(f"Metric {metric} is not supported!")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("NullDistortion cannot be called!")

    def populate_length_scale(
        self, hyperparameters: Dict, *args, **kwargs
    ) -> Dict:
        """
        A no-op placeholder function for NullDistortion objects.
        """
        return hyperparameters

    def get_opt_fn(self, *args, **kwargs):
        raise NotImplementedError(
            "NullDistortion cannot be used in optimization!"
        )

    def get_optim_params(self, *args, **kwargs):
        raise NotImplementedError(
            "NullDistortion cannot be used in optimization!"
        )
