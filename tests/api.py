import os
import sys

import numpy as np
import pickle as pkl

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.embed import embed_all
from MuyGPyS.examples.classify import (
    do_classify,
    example_lambdas,
    make_masks,
    do_uq,
)
from MuyGPyS.testing.test_utils import _basic_nn_kwarg_options


hardpath = "../data/star-gal/"

hardfiles = {
    "full": "galstar.pkl",
    "30": "embedded_30_galstar.pkl",
    "40": "embedded_40_galstar.pkl",
    "50": "embedded_50_galstar.pkl",
}


class StargalTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(StargalTest, cls).setUpClass()
        # with open(os.path.join(hardpath, hardfiles["full"]), "rb") as f:
        #     cls.train, cls.test = pkl.load(f)
        # with open(os.path.join(hardpath, hardfiles["30"]), "rb") as f:
        #     cls.embedded_30_train, cls.embedded_30_test = pkl.load(f)
        with open(os.path.join(hardpath, hardfiles["40"]), "rb") as f:
            cls.embedded_40_train, cls.embedded_40_test = pkl.load(f)
        # with open(os.path.join(hardpath, hardfiles["50"]), "rb") as f:
        #     cls.embedded_50_train, cls.embedded_50_test = pkl.load(f)

    @parameterized.parameters(
        (
            (
                nn,
                ed,
                ob,
                ub,
                uq,
                nn_kwargs,
                k_ta_dict[0],
                k_ta_dict[1],
                k_ta_dict[2],
            )
            for nn in [30]
            for ed in [40]
            for ob in [500]
            for ub in [500]
            # for uq in [example_lambdas]
            # for e in [True]
            for uq in [None, example_lambdas]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_ta_dict in (
                (
                    "matern",
                    0.97,
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                (
                    "rbf",
                    0.945,
                    {
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                # (
                #     "nngp",
                #     0.935,
                #     {
                #         "sigma_w_sq": 1.5,
                #         "sigma_b_sq": 1.0,
                #         "eps": 0.015,
                #         "sigma_sq": [1.0],
                #     },
                # ),
            )
        )
    )
    def test_classify_notrain_noembed(
        self,
        nn_count,
        embed_dim,
        opt_batch_size,
        uq_batch_size,
        uq_objectives,
        nn_kwargs,
        kern,
        target_accuracy,
        hyper_dict,
    ):

        self._do_classify_test_chassis(
            self.embedded_40_train,
            self.embedded_40_test,
            target_accuracy,
            nn_count,
            embed_dim,
            opt_batch_size,
            uq_batch_size,
            None,
            None,
            uq_objectives,
            nn_kwargs,
            kern,
            hyper_dict,
        )

    @parameterized.parameters(
        (
            (
                nn,
                ed,
                ob,
                ub,
                lm,
                uq,
                nn_kwargs,
                k_ta_dict[0],
                k_ta_dict[1],
                k_ta_dict[2],
            )
            for nn in [30]
            for ed in [40]
            for ob in [500]
            for ub in [500]
            # for lm in ["log"]
            # for uq in [example_lambdas]
            # for e in [True]
            for lm in ["log", "mse"]
            for uq in [None, example_lambdas]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_ta_dict in (
                (
                    "matern",
                    0.97,
                    {
                        # "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                (
                    "rbf",
                    0.945,
                    {
                        # "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": [1.0],
                    },
                ),
                # (
                #     "nngp",
                #     0.935,
                #     {
                #         "sigma_w_sq": 1.5,
                #         "sigma_b_sq": 1.0,
                #         "eps": 0.015,
                #         "sigma_sq": [1.0],
                #     },
                # ),
            )
        )
    )
    def test_classify_noembed(
        self,
        nn_count,
        embed_dim,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        uq_objectives,
        nn_kwargs,
        kern,
        target_accuracy,
        hyper_dict,
    ):

        self._do_classify_test_chassis(
            self.embedded_40_train,
            self.embedded_40_test,
            target_accuracy,
            nn_count,
            embed_dim,
            opt_batch_size,
            uq_batch_size,
            loss_method,
            None,
            uq_objectives,
            nn_kwargs,
            kern,
            hyper_dict,
        )

    def _do_classify_test_chassis(
        self,
        train,
        test,
        target_acc,
        nn_count,
        embed_dim,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        embed_method,
        uq_objectives,
        nn_kwargs,
        kern,
        hyper_dict,
    ):
        surrogate_predictions, predicted_labels, acc, masks = self._do_classify(
            train,
            test,
            nn_count,
            embed_dim,
            opt_batch_size,
            uq_batch_size,
            None,
            None,
            uq_objectives,
            nn_kwargs,
            kern,
            hyper_dict,
        )
        self.assertEqual(surrogate_predictions.shape, test["output"].shape)
        self.assertEqual(predicted_labels.shape, (test["output"].shape[0],))
        self.assertSequenceAlmostEqual(
            np.sum(surrogate_predictions, axis=1),
            np.zeros(surrogate_predictions.shape[0]),
        )
        self.assertSequenceAlmostEqual(
            np.unique(predicted_labels),
            np.unique(train["lookup"]),
        )
        self.assertGreaterEqual(acc, target_acc)
        if masks is not None:
            accuracy, uq = do_uq(predicted_labels, test, masks)
            self.assertEqual(
                masks.shape, (len(uq_objectives), test["output"].shape[0])
            )
            self.assertEqual(uq.shape, (len(uq_objectives), 3))
            # @NOTE[bwp] Should we do more to validate the uq? Expected ranges?
            # What about the first dimension `np.sum(mask)`, which records the
            # number of "ambiguous" prediction locations?
            # print(uq)
            for i in range(uq.shape[0] - 1):
                self.assertLessEqual(uq[i, 1], acc)
                self.assertGreaterEqual(uq[i, 2], acc)
            self.assertLessEqual(uq[-1, 1], 1.0)
            self.assertGreaterEqual(uq[-1, 2], target_acc)

    def _do_classify(
        self,
        train,
        test,
        nn_count,
        embed_dim,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        embed_method,
        uq_objectives,
        nn_kwargs,
        kern,
        hyper_dict,
    ):
        surrogate_predictions = do_classify(
            train,
            test,
            embed_method=None,
            opt_batch_size=opt_batch_size,
            uq_batch_size=uq_batch_size,
            embed_dim=embed_dim,
            nn_count=nn_count,
            kern=kern,
            hyper_dict=hyper_dict,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            verbose=False,
        )
        if uq_objectives is not None:
            surrogate_predictions, masks = surrogate_predictions
        else:
            masks = None
        predicted_labels = np.argmax(surrogate_predictions, axis=1)
        acc = np.mean(predicted_labels == np.argmax(test["output"], axis=1))
        predicted_labels = 2 * predicted_labels - 1
        return surrogate_predictions, predicted_labels, acc, masks


import sys
import os

if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
