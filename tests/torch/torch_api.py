# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import os
import sys

import pickle as pkl

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math.numpy as np
import MuyGPyS._src.math.torch as torch
from MuyGPyS import config
from MuyGPyS._src.math.torch import nn
from MuyGPyS._test.api import RegressionAPITest
from MuyGPyS.examples.muygps_torch import train_deep_kernel_muygps
from MuyGPyS.examples.muygps_torch import predict_model
from MuyGPyS.gp.hyperparameter import ScalarParam
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.torch import MuyGPs_layer
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.kernels import Matern

if config.state.torch_enabled is False:
    raise ValueError("Bad attempt to run torch-only code with torch disabled.")

if config.state.backend != "torch":
    raise ValueError(
        f"Bad attempt to run torch-only code in {config.state.backend} mode."
    )

if config.state.ftype != "32":
    raise ValueError(
        "Torch optimization only supports 32-bit values at this time."
    )

hardpath = "../data/"

stargal_dir = "star-gal/"

stargal_files = {
    "full": "galstar.pkl",
    "30": "embedded_30_galstar.pkl",
    "40": "embedded_40_galstar.pkl",
    "50": "embedded_50_galstar.pkl",
}

heaton_file = "heaton/sub_heaton.pkl"


# class SVDKMuyGPs_Star_Galaxy(nn.Module):
#     def __init__(
#         self,
#         muygps_model,
#         batch_indices,
#         batch_nn_indices,
#         batch_targets,
#         batch_nn_targets,
#     ):
#         super().__init__()
#         self.embedding = nn.Sequential(
#             nn.Linear(40, 30),
#             nn.Dropout(0.5),
#             nn.PReLU(1),
#             nn.Linear(30, 10),
#             nn.Dropout(0.5),
#             nn.PReLU(1),
#         )
#         self.muygps_model = muygps_model
#         self.batch_indices = batch_indices
#         self.batch_nn_indices = batch_nn_indices
#         self.batch_targets = batch_targets
#         self.batch_nn_targets = batch_nn_targets
#         self.GP_layer = MuyGPs_layer(
#             self.muygps_model,
#             self.batch_indices,
#             self.batch_nn_indices,
#             self.batch_targets,
#             self.batch_nn_targets,
#         )
#         self.deformation = self.GP_layer.deformation

#     def forward(self, x):
#         predictions = self.embedding(x)
#         predictions, variances = self.GP_layer(predictions)
#         return predictions, variances


# class StargalRegressTest(RegressionAPITest):
#     @classmethod
#     def setUpClass(cls):
#         super(StargalRegressTest, cls).setUpClass()
#         with open(
#             os.path.join(hardpath + stargal_dir, stargal_files["40"]), "rb"
#         ) as f:
#             cls.embedded_40_train, cls.embedded_40_test = pkl.load(f)
#             cls.embedded_40_train = {
#                 "input": torch.ndarray(cls.embedded_40_train["input"]),
#                 "output": torch.ndarray(cls.embedded_40_train["output"]),
#             }
#             cls.embedded_40_test = {
#                 "input": torch.ndarray(cls.embedded_40_test["input"]),
#                 "output": torch.ndarray(cls.embedded_40_test["output"]),
#             }

#     @parameterized.parameters(((nn, bs) for nn in [30] for bs in [500]))
#     def test_regress(
#         self,
#         nn_count,
#         batch_count,
#     ):
#         target_mse = 1.0
#         train = _balanced_subsample(self.embedded_40_train, 10000)
#         test = _balanced_subsample(self.embedded_40_test, 1000)

#         train_features = train["input"]
#         train_responses = train["output"]
#         test_features = test["input"]

#         nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="hnsw")
#         train_count, num_test_responses = train_responses.shape

#         batch_indices, batch_nn_indices = sample_batch(
#             nbrs_lookup, batch_count, train_count
#         )

#         batch_targets = train_responses[batch_indices, :]
#         batch_nn_targets = train_responses[batch_nn_indices, :]

#         model_kwargs = {
#             "kernel": Matern(
#                 smoothness=ScalarParam(1.5),
#                 deformation=Isotropy(
#                     metric=l2,
#                     length_scale=ScalarParam(7.2),
#                 ),
#             ),
#             "noise": HomoscedasticNoise(1e-5),
#         }

#         muygps_model = MuyGPS(**model_kwargs)

#         model = SVDKMuyGPs_Star_Galaxy(
#             muygps_model=muygps_model,
#             batch_indices=batch_indices,
#             batch_nn_indices=batch_nn_indices,
#             batch_targets=batch_targets,
#             batch_nn_targets=batch_nn_targets,
#         )

#         nbrs_struct, model_trained = train_deep_kernel_muygps(
#             model=model,
#             train_features=train_features,
#             train_responses=train_responses,
#             batch_indices=batch_indices,
#             nbrs_lookup=nbrs_lookup,
#             training_iterations=10,
#             optimizer_method=torch.optim.Adam,
#             learning_rate=1e-3,
#             scheduler_decay=0.95,
#             loss_function="lool",
#             update_frequency=1,
#         )

#         model_trained.eval()

#         predictions, variances = predict_model(
#             model=model_trained,
#             test_features=test_features,
#             train_features=train_features,
#             train_responses=train_responses,
#             nbrs_lookup=nbrs_struct,
#             nn_count=nn_count,
#         )

#         test_responses = test["output"]
#         mse_actual = (
#             np.sum(
#                 (
#                     predictions.squeeze().detach().numpy()
#                     - test_responses.squeeze().detach().numpy()
#                 )
#                 ** 2
#             )
#             / test_responses.shape[0]
#         )
#         self.assertLessEqual(mse_actual, target_mse)


class SVDKMuyGPs_Heaton(nn.Module):
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
            nn.Linear(2, 30),
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
        self.deformation = self.GP_layer.deformation

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
                "output": torch.squeeze(torch.ndarray(cls.train["output"])),
            }
            cls.test = {
                "input": torch.ndarray(cls.test["input"]),
                "output": torch.squeeze(torch.ndarray(cls.test["output"])),
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
        train_count = train_responses.shape[0]

        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )

        batch_targets = train_responses[batch_indices]
        batch_nn_targets = train_responses[batch_nn_indices]

        muygps_model = MuyGPS(
            Matern(
                smoothness=ScalarParam(1.5),
                deformation=Isotropy(
                    metric=l2,
                    length_scale=ScalarParam(1.0),
                ),
            ),
            noise=HomoscedasticNoise(1e-5),
        )

        model = SVDKMuyGPs_Heaton(
            muygps_model=muygps_model,
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
