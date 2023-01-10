# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
from typing import Callable

from MuyGPyS.gp.muygps import MuyGPS


def _scipy_optimize(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    raise NotImplementedError(
        f"Scipy optimization is not supported in MuyGPs PyTorch module."
    )


def _bayes_opt_optimize(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    raise NotImplementedError(
        f"Bayesian optimization is not supported in MuyGPs PyTorch module."
    )
