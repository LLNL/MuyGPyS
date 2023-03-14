# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenience wrapper for GP prediction from indices.
"""

import numpy as np

from typing import Tuple, Union

from MuyGPyS.gp.distance import make_regress_tensors, crosswise_distances
from MuyGPyS.gp import MuyGPS, MultivariateMuyGPS as MMuyGPS


def regress_from_indices(
    muygps: Union[MuyGPS, MMuyGPS],
    indices: np.ndarray,
    nn_indices: np.ndarray,
    test: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(muygps, MuyGPS):
        metric = muygps.kernel.metric
    else:
        metric = muygps.metric
    crosswise_tensor, pairwise_tensor, batch_nn_targets = make_regress_tensors(
        metric, indices, nn_indices, test, train, targets
    )
    if isinstance(muygps, MuyGPS):
        pairwise_tensor = muygps.kernel(pairwise_tensor)
        crosswise_tensor = muygps.kernel(crosswise_tensor)
    return muygps.regress(
        pairwise_tensor, crosswise_tensor, batch_nn_targets, **kwargs
    )


def fast_regress_from_indices(
    self,
    indices: np.ndarray,
    nn_indices: np.ndarray,
    test_features: np.ndarray,
    train_features: np.ndarray,
    closest_index: np.ndarray,
    coeffs_tensor: np.ndarray,
) -> np.ndarray:
    crosswise_dists = crosswise_distances(
        test_features,
        train_features,
        indices,
        nn_indices,
    )

    return self.fast_regress(
        crosswise_dists,
        coeffs_tensor[closest_index, :, :],
    )
