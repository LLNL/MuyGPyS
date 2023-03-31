# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
from MuyGPyS import config

if config.state.torch_enabled is False:
    raise ValueError(f"Bad attempt to run torch-only code with torch disabled.")

if config.state.backend != "torch":
    import warnings

    warnings.warn(
        f"Attempting to run torch-only code in {config.state.backend} mode. "
        f"Force-switching MuyGPyS into the torch backend."
    )
    config.update("muygpys_backend", "torch")

import os
import sys

import pickle as pkl

from absl.testing import absltest
from absl.testing import parameterized


config.parse_flags_with_absl()  # Affords option setting from CLI

import MuyGPyS._src.math.numpy as np
import MuyGPyS._src.math.torch as torch
from MuyGPyS import config
from MuyGPyS._src.math.torch import nn
from MuyGPyS._test.api import RegressionAPITest
from MuyGPyS._test.utils import _balanced_subsample
from MuyGPyS.examples.muygps_torch import train_deep_kernel_muygps
from MuyGPyS.examples.muygps_torch import predict_model
from MuyGPyS.gp.kernels import Hyperparameter
from MuyGPyS.gp.noise import HomoscedasticNoise, HeteroscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.torch.muygps_layer import MuyGPs_layer, MultivariateMuyGPs_layer

hardpath = "../data/"

stargal_dir = "star-gal/"

stargal_files = {
    "full": "galstar.pkl",
    "30": "embedded_30_galstar.pkl",
    "40": "embedded_40_galstar.pkl",
    "50": "embedded_50_galstar.pkl",
}

heaton_file = "heaton/sub_heaton.pkl"


class SVDKMuyGPs(nn.Module):
    def __init__(
        self,
        num_models,
        eps,
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
        self.eps = eps
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
        predictions, variances = self.GP_layer(predictions)
        return predictions, variances


class MultivariateStargalRegressTest(RegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(MultivariateStargalRegressTest, cls).setUpClass()
        with open(
            os.path.join(hardpath + stargal_dir, stargal_files["40"]), "rb"
        ) as f:
            cls.embedded_40_train, cls.embedded_40_test = pkl.load(f)
            cls.embedded_40_train = {
                "input": torch.ndarray(cls.embedded_40_train["input"]),
                "output": torch.ndarray(cls.embedded_40_train["output"]),
            }
            cls.embedded_40_test = {
                "input": torch.ndarray(cls.embedded_40_test["input"]),
                "output": torch.ndarray(cls.embedded_40_test["output"]),
            }

    @parameterized.parameters(((nn, bs) for nn in [30] for bs in [500]))
    def test_regress(
        self,
        nn_count,
        batch_count,
    ):
        target_mse = 1.0
        train = _balanced_subsample(self.embedded_40_train, 10000)
        test = _balanced_subsample(self.embedded_40_test, 1000)

        train_features = train["input"]
        train_responses = train["output"]
        test_features = test["input"]

        nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="hnsw")
        train_count, num_test_responses = train_responses.shape

        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )

        batch_targets = train_responses[batch_indices, :]
        batch_nn_targets = train_responses[batch_nn_indices, :]

        model = SVDKMuyGPs(
            num_models=num_test_responses,
            eps=[HomoscedasticNoise(1e-6)] * num_test_responses,
            nu=[Hyperparameter(0.5)] * num_test_responses,
            length_scale=[Hyperparameter(1.0)] * num_test_responses,
            batch_indices=batch_indices,
            batch_nn_indices=batch_nn_indices,
            batch_targets=batch_targets,
            batch_nn_targets=batch_nn_targets,
        )

        nbrs_struct, model_trained = train_deep_kernel_muygps(
            model=model,
            train_features=train_features,
            train_responses=train_responses,
            batch_indices=batch_indices,
            nbrs_lookup=nbrs_lookup,
            training_iterations=10,
            optimizer_method=torch.optim.Adam,
            learning_rate=1e-3,
            scheduler_decay=0.95,
            loss_function="lool",
            update_frequency=1,
        )

        model_trained.eval()

        predictions, variances = predict_model(
            model=model_trained,
            test_features=test_features,
            train_features=train_features,
            train_responses=train_responses,
            nbrs_lookup=nbrs_struct,
            nn_count=nn_count,
        )

        test_responses = test["output"]
        mse_actual = (
            np.sum(
                (
                    predictions.squeeze().detach().numpy()
                    - test_responses.squeeze().detach().numpy()
                )
                ** 2
            )
            / test_responses.shape[0]
        )
        self.assertLessEqual(mse_actual, target_mse)


class SVDKMuyGPs_Heaton(nn.Module):
    def __init__(
        self,
        eps,
        nu,
        length_scale,
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(2, 30),
            nn.Dropout(0.5),
            nn.PReLU(1),
            nn.Linear(30, 10),
            nn.Dropout(0.5),
            nn.PReLU(1),
        )
        self.eps = eps
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
        predictions, variances = self.GP_layer(predictions)
        return predictions, variances


class HeatonTest(RegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(HeatonTest, cls).setUpClass()
        with open(os.path.join(hardpath, heaton_file), "rb") as f:
            cls.train, cls.test = pkl.load(f)
            cls.train = {
                "input": torch.ndarray(cls.train["input"]),
                "output": torch.ndarray(cls.train["output"]),
            }
            cls.test = {
                "input": torch.ndarray(cls.test["input"]),
                "output": torch.ndarray(cls.test["output"]),
            }

    @parameterized.parameters(((nn, bs) for nn in [50] for bs in [500]))
    def test_regress(
        self,
        nn_count,
        batch_count,
    ):
        target_mse = 10.0

        train_features = self.train["input"]
        train_responses = self.train["output"]
        test_features = self.test["input"]

        nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="hnsw")
        train_count, num_test_responses = train_responses.shape

        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )

        batch_targets = train_responses[batch_indices, :]
        batch_nn_targets = train_responses[batch_nn_indices, :]

        model = SVDKMuyGPs_Heaton(
            eps=HomoscedasticNoise(1e-3),
            nu=Hyperparameter(0.5),
            length_scale=Hyperparameter(1.0),
            batch_indices=batch_indices,
            batch_nn_indices=batch_nn_indices,
            batch_targets=batch_targets,
            batch_nn_targets=batch_nn_targets,
        )

        nbrs_struct, model_trained = train_deep_kernel_muygps(
            model=model,
            train_features=train_features,
            train_responses=train_responses,
            batch_indices=batch_indices,
            nbrs_lookup=nbrs_lookup,
            training_iterations=10,
            optimizer_method=torch.optim.Adam,
            learning_rate=1e-4,
            scheduler_decay=0.95,
            loss_function="lool",
            update_frequency=1,
        )

        model_trained.eval()

        predictions, variances = predict_model(
            model=model_trained,
            test_features=test_features,
            train_features=train_features,
            train_responses=train_responses,
            nbrs_lookup=nbrs_struct,
            nn_count=nn_count,
        )

        test_responses = self.test["output"]
        mse_actual = (
            np.sum(
                (
                    predictions.squeeze().detach().numpy()
                    - test_responses.squeeze().detach().numpy()
                )
                ** 2
            )
            / test_responses.shape[0]
        )
        self.assertLessEqual(mse_actual, target_mse)


if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
