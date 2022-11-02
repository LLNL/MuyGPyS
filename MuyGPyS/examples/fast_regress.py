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

from MuyGPyS.examples.regress import (
    make_regressor,
    make_multivariate_regressor,
    _empirical_correlation,
    _empirical_covariance,
)


# build coefficients matrix, neighbor lookup
def make_fast_regressor(
    muygps: MuyGPS,
    nbrs_lookup: NN_Wrapper,
    train_features: np.ndarray,
    train_responses: np.ndarray,
) -> np.ndarray:
    num_training_samples, _ = train_features.shape
    nn_indices, _ = nbrs_lookup.get_batch_nns(
        np.arange(0, num_training_samples)
    )
    nn_indices = np.array(nn_indices).astype(int)
    precomputed_coefficients_matrix = muygps.build_fast_regress_coeffs(
        train_features, nn_indices, train_responses
    )
    return precomputed_coefficients_matrix


def make_fast_multivariate_regressor(
    muygps: MMuyGPS,
    nbrs_lookup: NN_Wrapper,
    train_features: np.ndarray,
    train_responses: np.ndarray,
) -> np.ndarray:
    num_training_samples, _ = train_features.shape
    nn_indices, _ = nbrs_lookup.get_batch_nns(
        np.arange(0, num_training_samples)
    )
    nn_indices = np.array(nn_indices).astype(int)
    precomputed_coefficients_matrix = muygps.build_fast_regress_coeffs(
        train_features, nn_indices, train_responses
    )
    return precomputed_coefficients_matrix


# choose between multivariate and univariate
def _decide_and_make_fast_regressor(
    muygps: MMuyGPS,
    nbrs_lookup: NN_Wrapper,
    train_features: np.ndarray,
    train_responses: np.ndarray,
) -> np.ndarray:
    if isinstance(muygps, MuyGPS):
        return make_fast_regressor(
            muygps, nbrs_lookup, train_features, train_responses
        )
    else:
        return make_fast_multivariate_regressor(
            muygps, nbrs_lookup, train_features, train_responses
        )


def _unpack():
    return


# workflow, return relevant structures
def do_fast_regress(
    muygps: MMuyGPS,
    nbrs_lookup: NN_Wrapper,
    train_features: np.ndarray,
    train_responses: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:

    return


# fast regression with timing
def fast_regress_any():
    return
