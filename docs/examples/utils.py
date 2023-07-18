# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


import numpy as np
import matplotlib.pyplot as plt

from MuyGPyS._test.gp import benchmark_sample, BenchmarkGP
from MuyGPyS.gp.distortion import IsotropicDistortion
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import mse_fn


class UnivariateSampler:
    def __init__(
        self,
        lb=-10.0,
        ub=10.0,
        data_count=5001,
        train_step=10,
        kernel=Matern(
            nu=ScalarHyperparameter(2.0),
            metric=IsotropicDistortion(
                "l2", length_scale=ScalarHyperparameter(1.0)
            ),
        ),
        eps=HomoscedasticNoise(1e-14),
        measurement_eps=HomoscedasticNoise(1e-5),
    ):
        self.data_count = data_count
        self.train_step = train_step
        self.x = np.linspace(lb, ub, data_count).reshape(data_count, 1)
        self.test_features = self.x[
            np.mod(np.arange(data_count), train_step) != 0, :
        ]
        self.train_features = self.x[::train_step, :]
        self.test_count, _ = self.test_features.shape
        self.train_count, _ = self.train_features.shape
        self.measurement_eps = measurement_eps
        self.gp = BenchmarkGP(kernel=kernel, eps=eps)
        self.vis_subset_size = 10
        self.mid = int(self.train_count / 2)

    def features(self):
        return self.train_features, self.test_features

    def sample(self):
        y = benchmark_sample(self.gp, self.x)
        self.test_responses = y[
            np.mod(np.arange(self.data_count), self.train_step) != 0, :
        ]
        self.train_responses = y[:: self.train_step, :] + np.random.normal(
            0, self.measurement_eps(), size=(self.train_count, 1)
        )
        return self.train_responses, self.test_responses

    def plot_sample(self):
        _, axes = plt.subplots(2, 1, figsize=(15, 11))

        axes[0].set_title("Sampled Curve", fontsize=24)
        axes[0].set_xlabel("Feature Domain", fontsize=20)
        axes[0].set_ylabel("Response Range", fontsize=20)
        axes[0].plot(
            self.train_features,
            self.train_responses,
            "k*",
            label="perturbed train response",
        )
        axes[0].plot(
            self.test_features,
            self.test_responses,
            "g-",
            label="test response",
        )
        axes[0].legend(fontsize=20)

        self.plot_target_sub(axes[1])

        plt.tight_layout()

        plt.show()

    def plot_results(
        self,
        scipy_predictions,
        scipy_confidence_intervals,
        bayes_predictions,
        bayes_confidence_intervals,
    ):
        _, axes = plt.subplots(3, 1, figsize=(15, 16))

        self.plot_target(axes[0])
        self.plot_model(
            axes[0], "Bayes", bayes_predictions, bayes_confidence_intervals
        )
        self.plot_target_sub(axes[1])
        self.plot_model_sub(
            axes[1], "scipy", scipy_predictions, scipy_confidence_intervals
        )
        self.plot_target_sub(axes[2])
        self.plot_model_sub(
            axes[2], "Bayes", bayes_predictions, bayes_confidence_intervals
        )

        plt.tight_layout()

        plt.show()

    def plot_target(self, ax):
        ax.set_title("Sampled Curve", fontsize=24)
        ax.set_xlabel("Feature Domain", fontsize=20)
        ax.set_ylabel("Response Range", fontsize=20)
        ax.plot(
            self.train_features,
            self.train_responses,
            "k*",
            label="perturbed train response",
        )
        ax.plot(
            self.test_features,
            self.test_responses,
            "g-",
            label="test response",
        )

    def plot_model(self, ax, name, predictions, confidence_intervals):
        confidence_intervals = confidence_intervals.reshape(self.test_count)
        ax.set_title(f"Sampled Curve with {name} optimized model", fontsize=24)
        ax.plot(
            self.test_features,
            predictions,
            "r--",
            label="test predictions",
        )
        ax.fill_between(
            self.test_features[:, 0],
            (predictions[:, 0] - confidence_intervals),
            (predictions[:, 0] + confidence_intervals),
            facecolor="red",
            alpha=0.25,
            label="95% Confidence Interval",
        )
        ax.legend(fontsize=20)

    def plot_target_sub(self, ax):
        ax.set_title("Sampled Curve (subset)", fontsize=24)
        ax.set_xlabel("Feature Domain", fontsize=20)
        ax.set_ylabel("Response Range", fontsize=20)
        ax.plot(
            self.train_features[self.mid : self.mid + self.vis_subset_size],
            self.train_responses[self.mid : self.mid + self.vis_subset_size],
            "k*",
            label="perturbed train response",
        )
        ax.plot(
            self.test_features[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            self.test_responses[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            "g-",
            label="test response",
        )

    def plot_model_sub(self, ax, name, predictions, confidence_intervals):
        confidence_intervals = confidence_intervals.reshape(self.test_count)
        ax.set_title(
            f"Sampled Curve (subset) with {name} optimized model", fontsize=24
        )
        ax.plot(
            self.test_features[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            predictions[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            "b--",
            label="test predictions",
        )
        ax.fill_between(
            self.test_features[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ][:, 0],
            (predictions[:, 0] - confidence_intervals)[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            (predictions[:, 0] + confidence_intervals)[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            facecolor="blue",
            alpha=0.25,
            label="95% Confidence Interval",
        )
        ax.legend(fontsize=20)


def print_results(
    name, targets, means, variances, confidence_intervals, coverage
):
    print(f"{name} results:")
    print(f"\tRMSE: {np.sqrt(mse_fn(means, targets))}")
    print(f"\tmean diagonal variance: {np.mean(variances)}")
    print(
        f"\tmean confidence interval size: {np.mean(confidence_intervals * 2)}"
    )
    print(f"\tcoverage: {coverage}")
