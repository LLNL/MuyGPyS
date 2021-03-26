import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import (
    sample_batch,
    sample_balanced_batch,
    full_filtered_batch,
)
from MuyGPyS.testing.test_utils import (
    BenchmarkGP,
    _optim_chassis,
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
    _basic_nn_kwarg_options,
)


class BatchTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, b, nn_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for b in [10000, 1000, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_batch(
        self, data_count, feature_count, nn_count, batch_count, nn_kwargs
    ):
        data = _make_gaussian_matrix(data_count, feature_count)
        nbrs_lookup = NN_Wrapper(data, nn_count, **nn_kwargs)
        indices, nn_indices = sample_batch(nbrs_lookup, batch_count, data_count)
        target_count = np.min((data_count, batch_count))
        self.assertEqual(indices.shape, (target_count,))
        self.assertEqual(nn_indices.shape, (target_count, nn_count))

    @parameterized.parameters(
        (
            (1000, f, r, nn, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_full_filtered_batch(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = full_filtered_batch(nbrs_lookup, data["lookup"])
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, ind in enumerate(indices):
            self.assertNotEqual(
                len(np.unique(data["lookup"][nn_indices[i, :]])), 1
            )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [10000, 1000, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["lookup"], batch_count
        )
        target_count = np.min((data_count, batch_count))
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, ind in enumerate(indices):
            self.assertNotEqual(
                len(np.unique(data["lookup"][nn_indices[i, :]])), 1
            )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch_lo_dist(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["lookup"], batch_count
        )
        target_count = np.min((data_count, batch_count))
        hist, _ = np.array(
            np.histogram(data["lookup"][indices], bins=response_count)
        )
        self.assertSequenceAlmostEqual(
            hist, (batch_count / response_count) * np.ones((response_count))
        )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [1000, 10000]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch_hi_dist(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["lookup"], batch_count
        )
        target_count = np.min((data_count, batch_count))
        hist, _ = np.array(
            np.histogram(data["lookup"][indices], bins=response_count)
        )
        self.assertGreater(
            np.mean(hist) + 0.1 * (target_count / response_count),
            target_count / response_count,
        )
        self.assertGreater(
            np.min(hist) + 0.45 * (target_count / response_count),
            target_count / response_count,
        )


class ObjectiveTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, r, nn, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_full_filtered_batch(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = full_filtered_batch(nbrs_lookup, data["lookup"])
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, ind in enumerate(indices):
            self.assertNotEqual(
                len(np.unique(data["lookup"][nn_indices[i, :]])), 1
            )


class BalancedBatchTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, r, nn, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_full_filtered_batch(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = full_filtered_batch(nbrs_lookup, data["lookup"])
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, ind in enumerate(indices):
            self.assertNotEqual(
                len(np.unique(data["lookup"][nn_indices[i, :]])), 1
            )


class GPSigmaSqBaselineTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1001,
                10,
                k_dict[0],
                k_dict[1],
            )
            for k_dict in (
                (
                    "matern",
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                    },
                ),
                (
                    "rbf",
                    {"length_scale": 1.5, "eps": 0.00001},
                ),
                (
                    "nngp",
                    {
                        "sigma_w_sq": 1.5,
                        "sigma_b_sq": 1.0,
                        "eps": 0.00001,
                    },
                ),
            )
        )
    )
    def test_baseline_sigma_sq_optim(
        self,
        data_count,
        its,
        kern,
        hyper_dict,
    ):
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2]
        sim_test["input"] = x[1::2]
        mean_sigma_sq = 0.0
        for i in range(its):
            gp = BenchmarkGP(kern=kern, **hyper_dict)
            gp.fit(sim_train["input"], sim_test["input"])
            y = gp.simulate()
            mean_sigma_sq += gp.get_sigma_sq(y)
        mean_sigma_sq /= its
        self.assertAlmostEqual(mean_sigma_sq, 1.0, 1)


class GPSigmaSqOptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1001,
                20,
                b,
                n,
                nn_kwargs,
                k_dict[0],
                k_dict[1],
            )
            for b in [250]
            for n in [34]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_dict in (
                (
                    "matern",
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                    },
                ),
                (
                    "rbf",
                    {"length_scale": 1.5, "eps": 0.00001},
                ),
                (
                    "nngp",
                    {
                        "sigma_w_sq": 1.5,
                        "sigma_b_sq": 1.0,
                        "eps": 0.00001,
                    },
                ),
            )
        )
    )
    def test_sigma_sq_optim(
        self,
        data_count,
        its,
        batch_count,
        nn_count,
        nn_kwargs,
        kern,
        hyper_dict,
    ):
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]
        mse = 0.0
        for i in range(its):
            gp = BenchmarkGP(kern=kern, **hyper_dict)
            gp.fit(sim_train["input"], sim_test["input"])
            y = gp.simulate()
            sim_train["output"] = y[:train_count].reshape(train_count, 1)
            sim_test["output"] = y[train_count:].reshape(test_count, 1)
            true_sigma_sq = gp.get_sigma_sq(y)
            global_sigmas, indiv_sigmas = _optim_chassis(
                sim_train,
                sim_train,
                nn_count,
                batch_count,
                kern=kern,
                hyper_dict=hyper_dict,
                nn_kwargs=nn_kwargs,
            )
            mse += (global_sigmas[0] - true_sigma_sq) ** 2
            # print(
            #     global_sigmas[0],
            #     true_sigma_sq,
            #     np.abs(global_sigmas[0] - true_sigma_sq),
            # )
        mse /= its
        self.assertAlmostEqual(mse, 0.0, 0)


class GPOptimTest(parameterized.TestCase):
    """
    @NOTE[bwp] We are seeing bad performance on recovering true hyperparameters
    for all settings aside from the `matern`/`nu` pairing using exact nearest
    neighbors, where we are able to find low mse. Is this due to the example
    data (a 1d curve), or a poor choice of true hyperparameters? I am not sure.
    Likely the 1d problem is the source of trouble concerning HNSW, as HNSW is
    designed for high dimensional large data problems.
    """

    @parameterized.parameters(
        (
            (
                1001,
                20,
                b,
                n,
                nn_kwargs,
                k_p_b_dict[0],
                k_p_b_dict[1],
                k_p_b_dict[2],
                k_p_b_dict[3],
            )
            for b in [250]
            for n in [34]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_p_b_dict in (
                (
                    "matern",
                    "nu",
                    (1e-5, 1.0),
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                # (
                #     "rbf",
                #     "length_scale",
                #     (1.0, 2.0),
                #     {
                #         "length_scale": 1.5,
                #         "eps": 0.00001,
                #         "sigma_sq": [1.0],
                #     },
                # ),
                # (
                #     "nngp",
                #     "sigma_w_sq",
                #     (0.5, 3.0),
                #     {
                #         "sigma_w_sq": 1.5,
                #         "sigma_b_sq": 1.0,
                #         "eps": 0.00001,
                #         "sigma_sq": [1.0],
                #     },
                # ),
            )
        )
    )
    def test_hyper_optim(
        self,
        data_count,
        its,
        batch_count,
        nn_count,
        nn_kwargs,
        kern,
        param,
        optim_bounds,
        hyper_dict,
    ):
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]
        mse = 0.0
        for i in range(its):
            gp = BenchmarkGP(kern=kern, **hyper_dict)
            gp.fit(sim_train["input"], sim_test["input"])
            y = gp.simulate()
            sim_train["output"] = y[:train_count].reshape(train_count, 1)
            sim_test["output"] = y[train_count:].reshape(test_count, 1)
            subtracted_dict = {
                k: hyper_dict[k] for k in hyper_dict if k != param
            }
            tru_param = hyper_dict[param]
            est_param = _optim_chassis(
                sim_train,
                sim_train,
                nn_count,
                batch_count,
                kern=kern,
                hyper_dict=subtracted_dict,
                optim_bounds={param: optim_bounds},
                nn_kwargs=nn_kwargs,
            )[0]
            mse += (est_param - tru_param) ** 2
            # print(
            #     est_param,
            #     tru_param,
            #     np.abs(est_param - tru_param),
            # )
        mse /= its
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(mse, 0.0, 1)


if __name__ == "__main__":
    absltest.main()
