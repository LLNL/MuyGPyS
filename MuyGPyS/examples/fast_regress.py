# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Resources and high-level API for a fast regression workflow.

:func:`~MuyGPyS.examples.regress.make_fast_regressor` is a high-level API for
creating and training :class:`MuyGPyS.gp.muygps.MuyGPS` objects for regression.
:func:`~MuyGPyS.examples.regress.make_multivariate_regressor` is a high-level
API for creating and training :class:`MuyGPyS.gp.muygps.MultivariateMuyGPS`
objects for regression.

:func:`~MuyGPyS.examples.regress.do_fast_regress` is a high-level api 
for executing a simple, generic regression workflow given data. 
It calls the maker APIs above and :func:`~MuyGPyS.examples.regress.regress_any`.
"""

import numpy as np

from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

from MuyGPyS.gp.distance import make_train_tensors
from MuyGPyS.optimize.chassis import optimize_from_tensors

from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.sigma_sq import (
    muygps_sigma_sq_optim,
    mmuygps_sigma_sq_optim,
)
