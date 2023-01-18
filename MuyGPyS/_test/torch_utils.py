# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
from torch import nn
from MuyGPyS.torch.muygps_layer import MuyGPs_layer, MultivariateMuyGPs_layer


class SVDKMuyGPs(nn.Module):
    def __init__(
        self,
        kernel_eps,
        nu,
        length_scale,
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(40, 30),
            nn.Dropout(0.5),
            nn.PReLU(1),
            nn.Linear(30, 10),
            nn.Dropout(0.5),
            nn.PReLU(1),
        )
        self.eps = kernel_eps
        self.nu = nu
        self.length_scale = length_scale
        self.batch_indices = batch_indices
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets
        self.GP_layer = MuyGPs_layer(
            self.eps,
            self.nu,
            self.length_scale,
            self.batch_indices,
            self.batch_nn_indices,
            self.batch_targets,
            self.batch_nn_targets,
        )

    def forward(self, x):
        predictions = self.embedding(x)
        predictions, variances, sigma_sq = self.GP_layer(predictions)
        return predictions, variances, sigma_sq


class SVDKMultivariateMuyGPs(nn.Module):
    def __init__(
        self,
        num_models,
        kernel_eps,
        nu,
        length_scale,
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(40, 30),
            nn.Dropout(0.5),
            nn.PReLU(1),
            nn.Linear(30, 10),
            nn.Dropout(0.5),
            nn.PReLU(1),
        )
        self.eps = kernel_eps
        self.nu = nu
        self.length_scale = length_scale
        self.batch_indices = batch_indices
        self.num_models = num_models
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets
        self.GP_layer = MultivariateMuyGPs_layer(
            self.num_models,
            self.eps,
            self.nu,
            self.length_scale,
            self.batch_indices,
            self.batch_nn_indices,
            self.batch_targets,
            self.batch_nn_targets,
        )

    def forward(self, x):
        predictions = self.embedding(x)
        predictions, variances, sigma_sq = self.GP_layer(predictions)
        return predictions, variances, sigma_sq
