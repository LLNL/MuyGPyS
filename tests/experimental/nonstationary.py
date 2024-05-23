# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np

from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.kernels import Matern, RBF
from MuyGPyS.gp.deformation import l2, Isotropy
from MuyGPyS.gp.hyperparameter import (
    Parameter,
    VectorParameter,
)
from MuyGPyS.gp.hyperparameter.experimental import (
    HierarchicalParameter,
    NamedHierarchicalParam,
    sample_knots,
)
from MuyGPyS.gp.tensors import batch_features_tensor
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch

from MuyGPyS._test.utils import (
    _check_ndarray,
    _make_gaussian_dict,
    _make_gaussian_data,
)


class HierarchicalNonstationaryHyperparameterTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (kernel,)
            for kernel in [
                RBF(),
                Matern(),
            ]
        )
    )
    def test_hierarchical_nonstationary_hyperparameter(
        self,
        kernel,
    ):
        knot_count = 10
        batch_count = 50
        train, test = _make_gaussian_data(
            train_count=knot_count,
            test_count=batch_count,
            feature_count=1000,
            response_count=1,
        )
        knot_features = train["input"]
        knot_values = VectorParameter(
            *[Parameter(x) for x in np.squeeze(train["output"])]
        )
        batch_features = test["input"]
        hyp = NamedHierarchicalParam(
            "custom_param_name",
            HierarchicalParameter(
                knot_features,
                knot_values,
                kernel,
            ),
        )
        hyperparameters = hyp(batch_features)
        _check_ndarray(
            self.assertEqual, hyperparameters, mm.ftype, shape=(batch_count,)
        )

    @parameterized.parameters(
        (
            (
                feature_count,
                type(high_level_kernel).__name__,
                deformation,
            )
            for feature_count in [2, 17]
            for knot_count in [10]
            for knot_features in [
                sample_knots(feature_count=feature_count, knot_count=knot_count)
            ]
            for knot_values in [
                VectorParameter(*[Parameter(i) for i in range(knot_count)]),
            ]
            for high_level_kernel in [RBF(), Matern()]
            for deformation in [
                Isotropy(
                    l2,
                    length_scale=HierarchicalParameter(
                        knot_features, knot_values, high_level_kernel
                    ),
                ),
                # Anisotropy(
                #     l2,
                #     VectorParameter(
                #         *[
                #             HierarchicalParameter(
                #                 knot_features,
                #                 knot_values,
                #                 high_level_kernel,
                #             )
                #             for _ in range(feature_count)
                #         ]
                #     ),
                # ),
            ]
        )
    )
    def test_hierarchical_nonstationary_rbf(
        self,
        feature_count,
        high_level_kernel,
        deformation,
    ):
        muygps = MuyGPS(
            kernel=RBF(deformation=deformation),
        )

        # prepare data
        data_count = 1000
        data = _make_gaussian_dict(
            data_count=data_count,
            feature_count=feature_count,
            response_count=1,
        )

        # neighbors and differences
        nn_count = 30
        nbrs_lookup = NN_Wrapper(
            data["input"], nn_count, nn_method="exact", algorithm="ball_tree"
        )
        batch_count = 200
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, data_count
        )
        (_, pairwise_diffs, _, _) = muygps.make_train_tensors(
            batch_indices,
            batch_nn_indices,
            data["input"],
            data["output"],
        )

        batch_features = batch_features_tensor(data["input"], batch_indices)

        Kin = muygps.kernel(pairwise_diffs, batch_features=batch_features)

        _check_ndarray(
            self.assertEqual,
            Kin,
            mm.ftype,
            shape=(batch_count, nn_count, nn_count),
        )


if __name__ == "__main__":
    absltest.main()
