import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from muyscans.gp.muygps import MuyGPS
from muyscans.neighbors import NN_Wrapper
from muyscans.testing.test_utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
    _basic_nn_kwarg_options,
)


class GPInitTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            "matern",
            {
                "nu": 0.38,
                "length_scale": 1.5,
                "eps": 0.00001,
                "sigma_sq": [0.98],
            },
        ),
        ("rbf", {"length_scale": 1.5, "eps": 0.00001, "sigma_sq": [0.98]}),
        (
            "nngp",
            {
                "sigma_w_sq": 1.5,
                "sigma_b_sq": 1.0,
                "eps": 0.00001,
                "sigma_sq": [0.98],
            },
        ),
    )
    def test_full_init(self, kern, hyper_dict):
        muygps = MuyGPS(kern=kern)
        unset_params = muygps.set_params(**hyper_dict)
        self.assertEmpty(unset_params)
        for key in hyper_dict:
            if key == "eps":
                self.assertEqual(hyper_dict[key], muygps.eps)
            elif key == "sigma_sq":
                self.assertEqual(hyper_dict[key], muygps.sigma_sq)
            else:
                self.assertEqual(hyper_dict[key], muygps.params[key])

    @parameterized.parameters(
        (
            "matern",
            "nu",
            {"length_scale": 1.5, "eps": 0.00001, "sigma_sq": [0.98]},
        ),
        (
            "matern",
            "length_scale",
            {"nu": 0.38, "eps": 0.00001, "sigma_sq": [0.98]},
        ),
        (
            "matern",
            "eps",
            {"nu": 0.38, "length_scale": 1.5, "sigma_sq": [0.98]},
        ),
        (
            "matern",
            "sigma_sq",
            {"nu": 0.38, "length_scale": 1.5, "eps": 0.00001},
        ),
        ("rbf", "length_scale", {"eps": 0.00001, "sigma_sq": [0.98]}),
        ("rbf", "eps", {"length_scale": 1.5, "sigma_sq": [0.98]}),
        ("rbf", "sigma_sq", {"length_scale": 1.5, "eps": 0.00001}),
        (
            "nngp",
            "sigma_w_sq",
            {"sigma_b_sq": 1.0, "eps": 0.00001, "sigma_sq": [0.98]},
        ),
        (
            "nngp",
            "sigma_b_sq",
            {"sigma_w_sq": 0.38, "eps": 0.00001, "sigma_sq": [0.98]},
        ),
        (
            "nngp",
            "eps",
            {"sigma_w_sq": 1.5, "sigma_b_sq": 1.0, "sigma_sq": [0.98]},
        ),
        (
            "nngp",
            "sigma_sq",
            {"sigma_w_sq": 1.5, "sigma_b_sq": 1.0, "eps": 0.00001},
        ),
    )
    def test_partial_init(self, kern, key, hyper_dict):
        muygps = MuyGPS(kern=kern)
        unset_params = muygps.set_params(**hyper_dict)
        self.assertEqual(len(unset_params), 1)
        self.assertIn(key, unset_params)


class GPMathTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1000,
                100,
                f,
                10,
                nn_kwargs,
                k,
            )
            for f in [100, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            for k in ["matern", "rbf", "nngp"]
        )
    )
    def test_tensor_shapes(
        self,
        train_count,
        test_count,
        feature_count,
        nn_count,
        nn_kwargs,
        kern,
    ):
        muygps = MuyGPS(kern=kern)
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        indices = [*range(test_count)]
        nn_indices = nbrs_lookup.get_nns(test)
        K, Kcross = muygps._compute_kernel_tensors(
            indices, nn_indices, test, train
        )
        self.assertEqual(K.shape, (test_count, nn_count, nn_count))
        self.assertEqual(Kcross.shape, (test_count, 1, nn_count))
        self.assertTrue(np.all(K >= 0.0))
        self.assertTrue(np.all(K <= 1.0))
        self.assertTrue(np.all(Kcross >= 0.0))
        self.assertTrue(np.all(Kcross <= 1.0))
        # Check that kernels are positive semidefinite
        for i in range(K.shape[0]):
            eigvals = np.linalg.eigvals(K[i, :, :])
            self.assertTrue(
                np.all(np.logical_or(eigvals >= 0.0, np.isclose(eigvals, 0.0)))
            )

    @parameterized.parameters(
        (
            (
                1000,
                100,
                f,
                r,
                10,
                nn_kwargs,
                k,
            )
            for f in [100, 1]
            for r in [10, 2, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            for k in ["matern", "rbf", "nngp"]
        )
    )
    def test_tensor_solve(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        kern,
    ):
        muygps = MuyGPS(kern=kern)
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)
        indices = [*range(test_count)]
        nn_indices = nbrs_lookup.get_nns(test["input"])
        K, Kcross = muygps._compute_kernel_tensors(
            indices, nn_indices, test["input"], train["input"]
        )
        solve = muygps._compute_solve(nn_indices, train["output"], K, Kcross)
        self.assertEqual(solve.shape, (test_count, response_count))
        for i in range(test_count):
            self.assertSequenceAlmostEqual(
                solve[i, :],
                Kcross[i, 0, :]
                @ np.linalg.solve(
                    K[i, :, :] + muygps.eps * np.eye(nn_count),
                    train["output"][nn_indices[i], :],
                ),
            )

    @parameterized.parameters(
        (
            (
                1000,
                100,
                f,
                r,
                10,
                nn_kwargs,
                k,
            )
            for f in [100, 1]
            for r in [10, 2, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            for k in ["matern", "rbf", "nngp"]
        )
    )
    def test_diagonal_variance(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        kern,
    ):
        muygps = MuyGPS(kern=kern)
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)
        indices = [*range(test_count)]
        nn_indices = nbrs_lookup.get_nns(test["input"])
        K, Kcross = muygps._compute_kernel_tensors(
            indices, nn_indices, test["input"], train["input"]
        )
        diagonal_variance = muygps._compute_diagonal_variance(K, Kcross)
        self.assertEqual(diagonal_variance.shape, (test_count,))
        for i in range(test_count):
            self.assertAlmostEqual(
                diagonal_variance[i],
                1.0
                - Kcross[i, 0, :]
                @ np.linalg.solve(
                    K[i, :, :] + muygps.eps * np.eye(nn_count), Kcross[i, 0, :]
                ),
            )
            self.assertGreater(diagonal_variance[i], 0.0)


class GPSigmaSqTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1000,
                f,
                r,
                10,
                nn_kwargs,
                k,
            )
            for f in [100, 1]
            for r in [10, 2, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            for k in ["matern", "rbf", "nngp"]
        )
    )
    def test_batch_sigma_sq_shapes(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        kern,
    ):
        muygps = MuyGPS(kern=kern)
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices = [*range(data_count)]
        nn_indices = nbrs_lookup.get_batch_nns(indices)
        muygps.sigma_sq_optim(
            indices, nn_indices, data["input"], data["output"]
        )
        self.assertEqual(muygps.sigma_sq.shape, (response_count,))
        for i in range(response_count):
            sigmas = muygps.get_sigma_sq(
                indices, nn_indices, data["input"], data["output"][:, i]
            )
            self.assertEqual(sigmas.shape, (data_count,))
            self.assertAlmostEqual(muygps.sigma_sq[i], np.mean(sigmas), 5)
        # print(sigmas.shape)


if __name__ == "__main__":
    absltest.main()
