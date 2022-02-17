# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np


def _mse_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    batch_count, response_count = predictions.shape
    squared_errors = np.sum((predictions - targets) ** 2)
    return squared_errors / (batch_count * response_count)
