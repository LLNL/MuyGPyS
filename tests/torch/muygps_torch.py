# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest

# from absl.testing import parameterized

# import MuyGPyS._src.math.numpy as np
# import MuyGPyS._src.math.torch as torch
from MuyGPyS import config

# from MuyGPyS._test.torch_utils import SVDKMuyGPs
# from MuyGPyS._test.utils import _check_ndarray, _make_gaussian_data
# from MuyGPyS.gp.kernels import Matern
# from MuyGPyS.gp import MuyGPS
# from MuyGPyS.gp.deformation import Isotropy, l2
# from MuyGPyS.gp.hyperparameter import ScalarParam
# from MuyGPyS.gp.noise import HomoscedasticNoise
# from MuyGPyS.optimize.batch import sample_batch
# from MuyGPyS.examples.muygps_torch import train_deep_kernel_muygps
# from MuyGPyS.examples.muygps_torch import predict_model
# from MuyGPyS.neighbors import NN_Wrapper


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


# class RegressTest(parameterized.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         super(RegressTest, cls).setUpClass()

#     @parameterized.parameters(
#         ((1000, 100, 40, 2, nn, bs) for nn in [30] for bs in [500])
#     )
#     def test_regress(
#         self,
#         train_count,
#         test_count,
#         feature_count,
#         response_count,
#         nn_count,
#         batch_count,
#     ):
#         target_mse = 3.0
#         train, test = _make_gaussian_data(
#             train_count, test_count, feature_count, response_count
#         )

#         train_features = train["input"]
#         train_responses = train["output"]
#         test_features = test["input"]
#         test_responses = test["output"]
#         _check_ndarray(self.assertEqual, train_features, torch.ftype)
#         _check_ndarray(self.assertEqual, train_responses, torch.ftype)
#         _check_ndarray(self.assertEqual, test_features, torch.ftype)
#         _check_ndarray(self.assertEqual, test_responses, torch.ftype)

#         nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="exact")
#         train_count, num_test_responses = train_responses.shape

#         batch_indices, batch_nn_indices = sample_batch(
#             nbrs_lookup, batch_count, train_count
#         )
#         _check_ndarray(self.assertEqual, batch_indices, torch.itype)
#         _check_ndarray(self.assertEqual, batch_nn_indices, torch.itype)

#         batch_targets = train_responses[batch_indices, :]
#         batch_nn_targets = train_responses[batch_nn_indices, :]
#         _check_ndarray(self.assertEqual, batch_targets, torch.ftype)
#         _check_ndarray(self.assertEqual, batch_nn_targets, torch.ftype)

#         model_smoothness = 0.5
#         model_length_scale = 1.0
#         measurement_noise = 1e-6

#         muygps_model = MuyGPS(
#             kernel=Matern(
#                 smoothness=ScalarParam(model_smoothness),
#                 deformation=Isotropy(
#                     metric=l2,
#                     length_scale=ScalarParam(model_length_scale),
#                 ),
#             ),
#             noise=HomoscedasticNoise(measurement_noise),
#         )

#         model = SVDKMuyGPs(
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

#         _check_ndarray(
#             self.assertEqual,
#             predictions,
#             torch.ftype,
#             shape=test_responses.shape,
#         )
#         _check_ndarray(
#             self.assertEqual,
#             variances,
#             torch.ftype,
#             shape=(test_count, response_count),
#         )
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


if __name__ == "__main__":
    absltest.main()
