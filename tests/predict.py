# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
from MuyGPyS.examples.classify import classify_any
from MuyGPyS.examples.two_class_classify_uq import (
    classify_two_class_uq,
    train_two_class_interval,
    example_lambdas,
    make_masks,
    do_uq,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import Isotropy, Anisotropy, l2
from MuyGPyS.gp.hyperparameter import ScalarParam, VectorParam
from MuyGPyS.gp.kernels import Matern, RBF
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import (
    get_balanced_batch,
)
from MuyGPyS._test.utils import (
    _make_gaussian_data,
    _basic_nn_kwarg_options,
)
from MuyGPyS._src.mpi_utils import _consistent_unchunk_tensor


if config.state.backend == "torch":
    raise ValueError("conventional optimization does not support torch.")


class ClassifyTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 200, 2, r, nn, nn_kwargs, k_kwargs)
            for r in [10, 1]
            for nn in [5, 50]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [100]
            # for r in [10]
            # for nn in [10]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for k_kwargs in (
                {
                    "kernel": Matern(
                        smoothness=ScalarParam(0.38),
                        deformation=Isotropy(l2, length_scale=ScalarParam(1.5)),
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": Matern(
                        smoothness=ScalarParam(0.38),
                        deformation=Anisotropy(
                            l2,
                            length_scale=VectorParam(
                                ScalarParam(1.5), ScalarParam(0.5)
                            ),
                        ),
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_classify_any(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        k_kwargs,
    ):
        muygps = MuyGPS(**k_kwargs)

        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)

        predictions, _ = classify_any(
            muygps,
            test["input"],
            train["input"],
            nbrs_lookup,
            train["output"],
        )
        predictions = _consistent_unchunk_tensor(predictions)
        self.assertEqual(predictions.shape, (test_count, response_count))


class ClassifyUQTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 200, 2, r, nn, b, nn_kwargs, k_kwargs)
            # for f in [100]
            # for r in [2]
            # for nn in [10]
            # for b in [200]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for r in [2]
            for nn in [5, 50]
            for b in [200]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                {
                    "kernel": Matern(
                        smoothness=ScalarParam(0.38),
                        deformation=Isotropy(l2, length_scale=ScalarParam(1.5)),
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": RBF(
                        deformation=Isotropy(l2, length_scale=ScalarParam(1.5))
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": Matern(
                        smoothness=ScalarParam(0.38),
                        deformation=Anisotropy(
                            l2,
                            length_scale=VectorParam(
                                ScalarParam(1.5), ScalarParam(0.5)
                            ),
                        ),
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": RBF(
                        deformation=Anisotropy(
                            l2,
                            length_scale=VectorParam(
                                ScalarParam(1.5), ScalarParam(0.5)
                            ),
                        )
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_classify_uq(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
        k_kwargs,
    ):
        muygps = MuyGPS(**k_kwargs)

        objective_count = len(example_lambdas)
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )
        train["output"] *= 2
        test["output"] *= 2
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)

        predictions, variances, _ = classify_two_class_uq(
            muygps,
            test["input"],
            train["input"],
            nbrs_lookup,
            train["output"],
        )

        predictions = _consistent_unchunk_tensor(predictions)
        variances = _consistent_unchunk_tensor(variances)
        self.assertEqual(predictions.shape, (test_count, response_count))
        self.assertEqual(variances.squeeze().shape, (test_count,))

        train_labels = np.argmax(train["output"], axis=1)
        indices, nn_indices = get_balanced_batch(
            nbrs_lookup,
            train_labels,
            batch_count,
        )

        cutoffs = train_two_class_interval(
            muygps,
            indices,
            nn_indices,
            train["input"],
            train["output"],
            train_labels,
            example_lambdas,
        )
        self.assertEqual(cutoffs.shape, (objective_count,))

        min_label = np.min(train["output"][0, :])
        max_label = np.max(train["output"][0, :])
        if min_label == 0.0 and max_label == 1.0:
            _ = np.argmax(predictions, axis=1)
        elif min_label == -1.0 and max_label == 1.0:
            _ = 2 * np.argmax(predictions, axis=1) - 1
        else:
            raise ("Unhandled label encoding min ({min_label}, {max_label})!")
        mid_value = (min_label + max_label) / 2

        masks = make_masks(predictions, cutoffs, variances, mid_value)
        self.assertEqual(masks.shape, (objective_count, test_count))

        acc, uq = do_uq(predictions, test["output"], masks)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertEqual(uq.shape, (objective_count, 3))


if __name__ == "__main__":
    absltest.main()
