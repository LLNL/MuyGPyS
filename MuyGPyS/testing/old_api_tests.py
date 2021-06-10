# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np
from absl.testing import parameterized

from MuyGPyS.examples.old_classify import (
    do_classify,
    example_lambdas,
    make_masks,
    do_uq,
)
from MuyGPyS.examples.old_regress import do_regress
from MuyGPyS.optimize.objective import mse_fn


class ClassifyAPITest(parameterized.TestCase):
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
            train=train,
            test=test,
            nn_count=nn_count,
            embed_dim=embed_dim,
            opt_batch_size=opt_batch_size,
            uq_batch_size=uq_batch_size,
            loss_method=loss_method,
            embed_method=embed_method,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            kern=kern,
            hyper_dict=hyper_dict,
        )
        self.assertEqual(surrogate_predictions.shape, test["output"].shape)
        self.assertEqual(predicted_labels.shape, (test["output"].shape[0],))
        self.assertSequenceAlmostEqual(
            np.sum(surrogate_predictions, axis=1),
            np.zeros(surrogate_predictions.shape[0]),
        )
        # There is almost certainly a better way to do this.
        if np.all(np.unique(train["output"]) == np.unique([-1, 1])):
            self.assertSequenceAlmostEqual(
                np.unique(predicted_labels),
                np.unique(train["output"]),
            )
        else:
            self.assertSequenceAlmostEqual(
                np.unique(predicted_labels),
                np.unique(np.argmax(train["output"], axis=1)),
            )
        print(f"obtained accuracy={acc}")
        self.assertGreaterEqual(acc, target_acc)
        if masks is not None:
            accuracy, uq = do_uq(surrogate_predictions, test["output"], masks)
            self.assertEqual(
                masks.shape, (len(uq_objectives), test["output"].shape[0])
            )
            self.assertEqual(uq.shape, (len(uq_objectives), 3))
            # @NOTE[bwp] Should we do more to validate the uq? Expected ranges?
            # What about the first dimension `np.sum(mask)`, which records the
            # number of "ambiguous" prediction locations?
            # print(uq)
            if kern == "matern":
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
            embed_method=embed_method,
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
        if np.all(np.unique(train["output"]) == np.unique([-1, 1])):
            predicted_labels = 2 * predicted_labels - 1
        return surrogate_predictions, predicted_labels, acc, masks


class RegressionAPITest(parameterized.TestCase):
    def _do_regress_test_chassis(
        self,
        train,
        test,
        target_mse,
        nn_count,
        embed_dim,
        batch_size,
        loss_method,
        variance_mode,
        embed_method,
        nn_kwargs,
        kern,
        hyper_dict,
    ):
        predictions, mse, variance, sigma_sq = self._do_regress(
            train,
            test,
            nn_count,
            embed_dim,
            batch_size,
            loss_method,
            variance_mode,
            embed_method,
            nn_kwargs,
            kern,
            hyper_dict,
        )
        self.assertEqual(predictions.shape, test["output"].shape)
        print(f"obtained mse={mse}")
        self.assertLessEqual(mse, target_mse)
        if variance is not None:
            test_count, response_count = test["output"].shape
            self.assertEqual(variance.shape, (test_count,))
            self.assertEqual(sigma_sq.shape, (response_count,))

    def _do_regress(
        self,
        train,
        test,
        nn_count,
        embed_dim,
        batch_size,
        loss_method,
        variance_mode,
        embed_method,
        nn_kwargs,
        kern,
        hyper_dict,
    ):
        predictions = do_regress(
            train,
            test,
            embed_method=embed_method,
            batch_size=batch_size,
            embed_dim=embed_dim,
            nn_count=nn_count,
            kern=kern,
            hyper_dict=hyper_dict,
            nn_kwargs=nn_kwargs,
            variance_mode=variance_mode,
            verbose=False,
        )
        if variance_mode is None:
            variance = None
            sigma_sq = None
        elif variance_mode == "diagonal":
            predictions, variance, sigma_sq = predictions
        else:
            raise ValueError(f"Variance mode {variance_mode} is not supported.")
        mse = mse_fn(predictions, test["output"])
        return predictions, mse, variance, sigma_sq