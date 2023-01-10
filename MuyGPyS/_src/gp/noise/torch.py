# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch


def _homoscedastic_perturb(K: torch.Tensor, eps: float) -> torch.Tensor:
    _, nn_count, _ = K.shape
    return K + eps * torch.eye(nn_count)
