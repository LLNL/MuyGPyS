# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np


def _homoscedastic_perturb(K: np.ndarray, eps: float) -> np.ndarray:
    _, nn_count, _ = K.shape
    return K + eps * np.eye(nn_count)
