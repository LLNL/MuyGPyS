# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
import torch
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
import torch
from torch import nn
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


config.parse_flags_with_absl()  # Affords option setting from CLI

from MuyGPyS.gp.distance import (
    make_train_tensors,
    make_regress_tensors,
    pairwise_distances,
    crosswise_distances,
)
from MuyGPyS.gp.muygps import MuyGPS

from MuyGPyS._test.utils import (
    _make_gaussian_dict,
    _make_gaussian_data,
    _make_gaussian_matrix,
    _basic_nn_kwarg_options,
    _basic_opt_method_and_kwarg_options,
    _balanced_subsample,
    _basic_nn_kwarg_options,
    _basic_opt_method_and_kwarg_options,
)

from MuyGPyS import config
from MuyGPyS.torch.muygps_layer import MuyGPs_layer, MultivariateMuyGPs_layer
from MuyGPyS._src.optimize.loss import _lool_fn as lool_fn
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.examples.muygps_torch import train_deep_kernel_muygps
from MuyGPyS.examples.muygps_torch import predict_model
from MuyGPyS.neighbors import NN_Wrapper

from MuyGPyS._test.torch_utils import SVDKMuyGPs, SVDKMultivariateMuyGPs


class RegressTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(RegressTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (1000, 100, 40, 2, nn, bs, vm)
            for nn in [30]
            for bs in [500]
            for vm in [None, "diagonal"]
        )
    )
    def test_regress(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        variance_mode,
    ):
        target_mse = 3.0
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )

        if variance_mode is None:
            sigma_method = None
            apply_sigma_sq = False
        else:
            sigma_method = "analytic"
            apply_sigma_sq = True

        train_features = train["input"]
        train_responses = train["output"]
        test_features = test["input"]

        nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="exact")
        train_count, num_test_responses = train_responses.shape

        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )

        batch_indices, batch_nn_indices = batch_indices.astype(
            np.int64
        ), batch_nn_indices.astype(np.int64)
        batch_indices, batch_nn_indices = torch.from_numpy(
            batch_indices
        ), torch.from_numpy(batch_nn_indices)

        batch_targets = train_responses[batch_indices, :]
        batch_nn_targets = train_responses[batch_nn_indices, :]

        batch_targets = torch.from_numpy(
            train_responses[batch_indices, :]
        ).float()
        batch_nn_targets = torch.from_numpy(
            train_responses[batch_nn_indices, :]
        ).float()

        model = SVDKMuyGPs(
            kernel_eps=1e-6,
            nu=1 / 2,
            length_scale=1.0,
            batch_indices=batch_indices,
            batch_nn_indices=batch_nn_indices,
            batch_targets=batch_targets,
            batch_nn_targets=batch_nn_targets,
        )

        train_features = torch.from_numpy(train_features).float()
        train_responses = torch.from_numpy(train_responses).float()

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

        test_features = torch.from_numpy(test_features).float()
        model_trained.eval()

        predictions, variances, sigma_sq = predict_model(
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
                    - test_responses.squeeze()
                )
                ** 2
            )
            / test_responses.shape[0]
        )
        self.assertEqual(predictions.shape, test_responses.shape)
        self.assertEqual(variances.shape, (test_count, response_count))
        self.assertEqual(sigma_sq.shape, torch.Size([num_test_responses]))
        self.assertLessEqual(mse_actual, target_mse)


class MultivariateRegressTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(MultivariateRegressTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (1000, 100, 40, 2, nn, bs, vm)
            for nn in [30]
            for bs in [500]
            for vm in [None, "diagonal"]
        )
    )
    def test_regress(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        variance_mode,
    ):
        target_mse = 3.0
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )

        if variance_mode is None:
            sigma_method = None
            apply_sigma_sq = False
        else:
            sigma_method = "analytic"
            apply_sigma_sq = True

        train_features = train["input"]
        train_responses = train["output"]
        test_features = test["input"]

        nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="exact")
        train_count, num_test_responses = train_responses.shape

        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )

        batch_indices, batch_nn_indices = batch_indices.astype(
            np.int64
        ), batch_nn_indices.astype(np.int64)
        batch_indices, batch_nn_indices = torch.from_numpy(
            batch_indices
        ), torch.from_numpy(batch_nn_indices)

        batch_targets = train_responses[batch_indices, :]
        batch_nn_targets = train_responses[batch_nn_indices, :]

        batch_targets = torch.from_numpy(
            train_responses[batch_indices, :]
        ).float()
        batch_nn_targets = torch.from_numpy(
            train_responses[batch_nn_indices, :]
        ).float()

        model = SVDKMultivariateMuyGPs(
            num_models=num_test_responses,
            kernel_eps=1e-6 * torch.ones(num_test_responses),
            nu=1 / 2 * torch.ones(num_test_responses),
            length_scale=1.0 * torch.ones(num_test_responses),
            batch_indices=batch_indices,
            batch_nn_indices=batch_nn_indices,
            batch_targets=batch_targets,
            batch_nn_targets=batch_nn_targets,
        )

        train_features = torch.from_numpy(train_features).float()
        train_responses = torch.from_numpy(train_responses).float()

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

        test_features = torch.from_numpy(test_features).float()
        model_trained.eval()

        predictions, variances, sigma_sq = predict_model(
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
                    - test_responses.squeeze()
                )
                ** 2
            )
            / test_responses.shape[0]
        )
        self.assertEqual(predictions.shape, test_responses.shape)
        self.assertEqual(variances.shape, (test_count, response_count))
        self.assertEqual(sigma_sq.shape, torch.Size([num_test_responses]))
        self.assertLessEqual(mse_actual, target_mse)


if __name__ == "__main__":
    absltest.main()
