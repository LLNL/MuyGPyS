import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from muyscans.examples.classify import example_lambdas, make_masks, do_uq
from muyscans.gp.lkgp import LKGP
from muyscans.neighbors import NN_Wrapper
from muyscans.optimize.batch import (
    get_classify_batch,
    sample_batch,
    sample_balanced_batch,
    full_filtered_batch,
)
from muyscans.optimize.objective import (
    cross_entropy_fn,
    mse_fn,
    loo_crossval,
)
from muyscans.predict import (
    classify_any,
    classify_two_class_uq,
)
from muyscans.testing.test_utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
)
from muyscans.uq import (
    train_two_class_interval,
)


class ClassifyTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 200, f, r, nn, e, k_dict[0], k_dict[1])
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for e in [True, False]
            for k_dict in (
                (
                    "matern",
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                (
                    "rbf",
                    {
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                (
                    "nngp",
                    {
                        "sigma_w_sq": 1.5,
                        "sigma_b_sq": 1.0,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
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
        is_exact,
        kern,
        hyper_dict,
    ):
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, is_exact)
        lkgp = LKGP(kern=kern)
        lkgp.set_params(**hyper_dict)

        predictions, _ = classify_any(
            lkgp,
            test["input"],
            train["input"],
            nbrs_lookup,
            train["output"],
            nn_count,
        )
        self.assertEqual(predictions.shape, (test_count, response_count))

    # @parameterized.parameters(
    #     (
    #         (1000, 200, f, nn, e, k_dict[0], k_dict[1])
    #         # for f in [100]
    #         # for nn in [10]
    #         # for e in [True]
    #         for f in [100, 10, 2]
    #         for nn in [5, 10, 100]
    #         for e in [True, False]
    #         for k_dict in (
    #             (
    #                 "matern",
    #                 {
    #                     "nu": 0.38,
    #                     "length_scale": 1.5,
    #                     "eps": 0.00001,
    #                     "sigma_sq": [1.0],
    #                 },
    #             ),
    #             (
    #                 "rbf",
    #                 {
    #                     "length_scale": 1.5,
    #                     "eps": 0.00001,
    #                     "sigma_sq": [1.0],
    #                 },
    #             ),
    #             (
    #                 "nngp",
    #                 {
    #                     "sigma_w_sq": 1.5,
    #                     "sigma_b_sq": 1.0,
    #                     "eps": 0.00001,
    #                     "sigma_sq": [1.0],
    #                 },
    #             ),
    #         )
    #     )
    # )
    # def test_classify_uq(
    #     self,
    #     train_count,
    #     test_count,
    #     feature_count,
    #     nn_count,
    #     is_exact,
    #     kern,
    #     hyper_dict,
    # ):
    #     response_count = 2
    #     train, test = _make_gaussian_data(
    #         train_count,
    #         test_count,
    #         feature_count,
    #         response_count,
    #         categorical=True,
    #     )
    #     nbrs_lookup = NN_Wrapper(train["input"], nn_count, is_exact)
    #     lkgp = LKGP(kern=kern)
    #     lkgp.set_params(**hyper_dict)

    #     predictions, variances, _ = classify_two_class_uq(
    #         lkgp,
    #         test["input"],
    #         train["input"],
    #         nbrs_lookup,
    #         train["output"],
    #         nn_count,
    #     )
    #     self.assertEqual(predictions.shape, (test_count, response_count))
    #     self.assertEqual(variances.shape, (test_count,))

    @parameterized.parameters(
        (
            (1000, 200, f, nn, b, e, k_dict[0], k_dict[1])
            # for f in [100]
            # for nn in [10]
            # for b in [200]
            # for e in [True]
            for f in [100, 10, 2]
            for nn in [5, 10, 100]
            for b in [200]
            for e in [True, False]
            for k_dict in (
                (
                    "matern",
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                (
                    "rbf",
                    {
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                (
                    "nngp",
                    {
                        "sigma_w_sq": 1.5,
                        "sigma_b_sq": 1.0,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
            )
        )
    )
    def test_classify_uq(
        self,
        train_count,
        test_count,
        feature_count,
        nn_count,
        batch_size,
        is_exact,
        kern,
        hyper_dict,
    ):
        response_count = 2
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
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, is_exact)
        lkgp = LKGP(kern=kern)
        lkgp.set_params(**hyper_dict)

        predictions, variances, _ = classify_two_class_uq(
            lkgp,
            test["input"],
            train["input"],
            nbrs_lookup,
            train["output"],
            nn_count,
        )

        self.assertEqual(predictions.shape, (test_count, response_count))
        self.assertEqual(variances.shape, (test_count,))

        train_lookup = np.argmax(train["output"], axis=1)
        indices, nn_indices = get_classify_batch(
            nbrs_lookup,
            train_lookup,
            batch_size,
            train_count,
        )

        cutoffs = train_two_class_interval(
            lkgp,
            indices,
            nn_indices,
            train["input"],
            train["output"],
            train_lookup,
            example_lambdas,
        )
        self.assertEqual(cutoffs.shape, (objective_count,))

        min_label = np.min(train["output"][0, :])
        max_label = np.max(train["output"][0, :])
        if min_label == 0.0 and max_label == 1.0:
            predicted_labels = np.argmax(predictions, axis=1)
        elif min_label == -1.0 and max_label == 1.0:
            predicted_labels = 2 * np.argmax(predictions, axis=1) - 1
        else:
            raise ("Unhandled label encoding min ({min_label}, {max_label})!")
        mid_value = (min_label + max_label) / 2

        masks = make_masks(predictions, cutoffs, variances, mid_value)
        self.assertEqual(masks.shape, (objective_count, test_count))

        acc, uq = do_uq(predicted_labels, test, masks)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertEqual(uq.shape, (objective_count, 3))


if __name__ == "__main__":
    absltest.main()
