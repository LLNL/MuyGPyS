# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.math.torch import nn
from MuyGPyS.torch import MuyGPs_layer, MultivariateMuyGPs_layer


class SVDKMuyGPs(nn.Module):
    def __init__(
        self,
        muygps_model,
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
        self.muygps_model = muygps_model
        self.batch_indices = batch_indices
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets
        self.GP_layer = MuyGPs_layer(
            self.muygps_model,
            self.batch_indices,
            self.batch_nn_indices,
            self.batch_targets,
            self.batch_nn_targets,
        )

    def forward(self, x):
        predictions = self.embedding(x)
        predictions, variances = self.GP_layer(predictions)
        return predictions, variances


class SVDKMultivariateMuyGPs(nn.Module):
    def __init__(
        self,
        multivariate_muygps_model,
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(40, 30),
            nn.Dropout(0.5),
            nn.ELU(1),
            nn.Linear(30, 10),
            nn.Dropout(0.5),
            nn.ELU(1),
        )
        self.multivariate_muygps_model = multivariate_muygps_model
        self.batch_indices = batch_indices
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets
        self.GP_layer = MultivariateMuyGPs_layer(
            self.multivariate_muygps_model,
            self.batch_indices,
            self.batch_nn_indices,
            self.batch_targets,
            self.batch_nn_targets,
        )

    def forward(self, x):
        predictions = self.embedding(x)
        predictions, variances = self.GP_layer(predictions)
        return predictions, variances
