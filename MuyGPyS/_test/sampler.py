# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MuyGPyS._test.gp import benchmark_sample, BenchmarkGP
from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.hyperparameter import ScalarParam
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import mse_fn


class SamplerBase:
    def __init__(self, data_count, train_ratio=0.1, seed=0):
        rng = np.random.default_rng(seed=seed)
        interval_count = int(data_count * train_ratio)
        interval = int(data_count / interval_count)
        sfl = rng.permutation(np.arange(data_count))
        self._train_mask = np.zeros(data_count, dtype=bool)
        for i in range(interval_count):
            idx = np.random.choice(sfl[i * interval : (i + 1) * interval])
            self._train_mask[idx] = True
        self._test_mask = np.invert(self._train_mask)
        self._title_fsize = 18
        self._axis_fsize = 14
        self._legend_fsize = 12


class UnivariateSampler(SamplerBase):
    def __init__(
        self,
        data_count=5001,
        train_ratio=0.1,
        seed=0,
        view_lb=0.5,
        view_ub=0.6,
        kernel=Matern(
            smoothness=ScalarParam(2.0),
            deformation=Isotropy(l2, length_scale=ScalarParam(1.0)),
        ),
        noise=HomoscedasticNoise(1e-14),
        measurement_noise=HomoscedasticNoise(1e-5),
    ):
        super().__init__(data_count, train_ratio=train_ratio, seed=seed)
        self.data_count = data_count
        self.x = np.linspace(0, 1, data_count).reshape(data_count, 1)
        self.test_features = self.x[self._test_mask, :]
        self.train_features = self.x[self._train_mask, :]
        self.test_count, _ = self.test_features.shape
        self.train_count, _ = self.train_features.shape
        self.measurement_noise = measurement_noise
        self.gp = BenchmarkGP(kernel=kernel, noise=noise)
        self.train_interval = self.get_interval(
            view_lb, view_ub, self.train_features
        )
        self.test_interval = self.get_interval(
            view_lb, view_ub, self.test_features
        )
        self._target_color = "#7570b3"
        self._predict_colors = ["#d95f02", "#1b9e77"]

    def features(self):
        return self.train_features, self.test_features

    def sample(self):
        y = benchmark_sample(self.gp, self.x)
        self.test_responses = y[self._test_mask, :]
        self.train_responses = y[self._train_mask, :] + np.random.normal(
            0, self.measurement_noise(), size=(self.train_count, 1)
        )
        return self.train_responses, self.test_responses

    def plot_sample(self):
        _, axes = plt.subplots(2, 1, figsize=(8, 7))

        axes[0].set_title("Sampled Curve", fontsize=self._title_fsize)
        axes[0].set_xlabel("Feature Domain", fontsize=self._axis_fsize)
        axes[0].set_ylabel("Response Range", fontsize=self._axis_fsize)
        axes[0].plot(
            self.train_features,
            self.train_responses,
            "k*",
            label="perturbed train response",
        )
        axes[0].plot(
            self.test_features,
            self.test_responses,
            "-",
            color=self._target_color,
            label="test response",
        )
        axes[0].legend(fontsize=self._legend_fsize)

        self.plot_target_sub(axes[1])

        plt.tight_layout()

        plt.show()

    def plot_results(
        self,
        *args,
    ):
        _, axes = plt.subplots(2, 1, figsize=(8, 8))

        self.plot_target(axes[0])
        self.plot_target_sub(axes[1])
        for i, (name, predictions, confidence_intervals) in enumerate(args):
            # if i == 0:
            self.plot_model(
                axes[0],
                name,
                predictions,
                confidence_intervals,
                color=self._predict_colors[i],
            )

            self.plot_model_sub(
                axes[1],
                name,
                predictions,
                confidence_intervals,
                color=self._predict_colors[i],
            )

        plt.tight_layout()

        plt.show()

    def plot_target(self, ax):
        ax.set_title("Sampled Curve", fontsize=self._title_fsize)
        ax.set_xlabel("Feature Domain", fontsize=self._axis_fsize)
        ax.set_ylabel("Response Range", fontsize=self._axis_fsize)
        ax.plot(
            self.train_features,
            self.train_responses,
            "k*",
            label="perturbed train response",
        )
        ax.plot(
            self.test_features,
            self.test_responses,
            "-",
            color=self._target_color,
            label="test response",
        )

    def plot_model(
        self,
        ax,
        name,
        predictions,
        confidence_intervals,
        color=None,
    ):
        if color is None:
            color = self._predict_colors[0]
        confidence_intervals = confidence_intervals.reshape(self.test_count)
        ax.set_title(
            f"Sampled Curve with {name} model",
            fontsize=self._title_fsize,
        )
        ax.plot(
            self.test_features,
            predictions,
            "--",
            color=color,
            label=f"{name} predictions",
        )
        ax.fill_between(
            self.test_features[:, 0],
            (predictions[:, 0] - confidence_intervals),
            (predictions[:, 0] + confidence_intervals),
            facecolor=color,
            alpha=0.25,
            label=f"{name} 95% Confidence Interval",
        )
        ax.legend(fontsize=self._legend_fsize)

    def get_interval(self, lb, ub, ys):
        return np.array(
            [i for i, x in enumerate(ys) if x >= lb and x <= ub], dtype=int
        )

    def plot_target_sub(self, ax):
        ax.set_title("Sampled Curve (subset)", fontsize=20)
        ax.set_xlabel("Feature Domain", fontsize=16)
        ax.set_ylabel("Response Range", fontsize=16)
        ax.plot(
            self.train_features[self.train_interval],
            self.train_responses[self.train_interval],
            "k*",
            label="perturbed train response",
        )
        ax.plot(
            self.test_features[self.test_interval],
            self.test_responses[self.test_interval],
            "-",
            color=self._target_color,
            label="test response",
        )

    def plot_model_sub(
        self, ax, name, predictions, confidence_intervals, color=None
    ):
        if color is None:
            color = self._predict_colors[0]
        confidence_intervals = confidence_intervals.reshape(self.test_count)
        ax.set_title(
            f"Sampled Curve (subset) with {name} model",
            fontsize=self._title_fsize,
        )
        ax.plot(
            self.test_features[self.test_interval],
            predictions[self.test_interval],
            "--",
            color=color,
            label=f"{name} predictions",
        )
        ax.fill_between(
            self.test_features[self.test_interval][:, 0],
            (predictions[:, 0] - confidence_intervals)[self.test_interval],
            (predictions[:, 0] + confidence_intervals)[self.test_interval],
            facecolor=color,
            alpha=0.25,
            label=f"{name} 95% Confidence Interval",
        )
        ax.legend(fontsize=self._legend_fsize)


class UnivariateSampler2D(SamplerBase):
    def __init__(
        self,
        points_per_dim=60,
        train_ratio=10,
        kernel=Matern(
            smoothness=ScalarParam(2.0),
            deformation=Isotropy(l2, length_scale=ScalarParam(1.0)),
        ),
        noise=HomoscedasticNoise(1e-14),
        measurement_noise=HomoscedasticNoise(1e-5),
    ):
        self.points_per_dim = points_per_dim
        self.data_count = self.points_per_dim**2
        super().__init__(self.data_count, train_ratio)
        x = np.linspace(0, 1, points_per_dim)
        xx, yy = np.meshgrid(x, x)
        self.xs = np.array(
            [
                [xx[i, j], yy[i, j]]
                for i in range(points_per_dim)
                for j in range(points_per_dim)
            ]
        )
        self.test_features = self.xs[self._test_mask, :]
        self.train_features = self.xs[self._train_mask, :]
        self.test_count, _ = self.test_features.shape
        self.train_count, _ = self.train_features.shape
        self.measurement_noise = measurement_noise
        self.gp = BenchmarkGP(kernel=kernel, noise=noise)

    def features(self):
        return self.train_features, self.test_features

    def sample(self):
        self.ys = benchmark_sample(self.gp, self.xs)
        self.test_responses = self.ys[self._test_mask, :]
        self.train_responses = self.ys[self._train_mask, :] + np.random.normal(
            0, self.measurement_noise(), size=(self.train_count, 1)
        )
        return self.train_responses, self.test_responses

    def plot_sample(self):
        fig, axes = plt.subplots(1, 3, figsize=(19, 4))

        vmin = np.nanmin(self.ys)
        vmax = np.nanmax(self.ys)

        self._label_ax(axes[0], "Sampled Surface")
        im0 = axes[0].imshow(
            self._make_im(self.ys, mode="all"), vmin=vmin, vmax=vmax
        )

        self._label_ax(axes[1], "Training Points")
        axes[1].imshow(
            self._make_im(self.train_responses, mode="train"),
            vmin=vmin,
            vmax=vmax,
        )

        self._label_ax(axes[2], "Testing Points")
        axes[2].imshow(self._make_im(self.test_responses), vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=axes.ravel().tolist())

        plt.show()

    def _make_im(self, array, mode="test", range=False, add_inf=True):
        im = np.zeros(self.data_count)
        if mode == "test":
            im[self._test_mask, None] = array
            if add_inf is True:
                im[self._train_mask] = -np.inf
        elif mode == "train":
            im[self._train_mask, None] = array
            if add_inf is True:
                im[self._test_mask] = -np.inf
        else:
            im[:, None] = array
        if range is False:
            return im.reshape(self.points_per_dim, self.points_per_dim)
        else:
            vmin = np.nanmin(array)
            vmax = np.nanmax(array)
            return (
                im.reshape(self.points_per_dim, self.points_per_dim),
                vmin,
                vmax,
            )

    def plot_predictions(self, *args):
        pred_count = len(args)
        fig, axes = plt.subplots(
            1, pred_count + 1, figsize=(4 * (pred_count + 1), 4)
        )

        test_im, vmin, vmax = self._make_im(self.test_responses, range=True)
        pred_ims = list()
        names = list()
        for name, predictions in args:
            pred_im, pvmin, pvmax = self._make_im(predictions, range=True)
            pred_ims.append(pred_im)
            names.append(name)
            vmin = np.min([vmin, pvmin])
            vmax = np.max([vmax, pvmax])

        self._label_ax(axes[0], "Expected Surface")
        im0 = axes[0].imshow(
            test_im,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        fig.colorbar(im0, ax=axes[0])

        for i, pred_im in enumerate(pred_ims):
            self._label_ax(axes[i + 1], f"{names[i]} Surface")
            im1 = axes[i + 1].imshow(
                pred_im,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            fig.colorbar(im1, ax=axes[i + 1])

        plt.tight_layout()
        plt.show()

    def _get_images(self, predictions, confidence_intervals):
        resl_im, resl_min, resl_max = self._make_im(
            self.test_responses - predictions, range=True
        )
        conf_im, _, conf_mag = self._make_im(confidence_intervals, range=True)
        covr_im, covr_min, covr_max = self._make_im(
            np.abs(self.test_responses - predictions) - confidence_intervals,
            range=True,
        )

        resl_mag = np.max([np.abs(resl_min), np.abs(resl_max)])
        covr_mag = np.max([np.abs(covr_min), np.abs(covr_max)])
        return resl_im, conf_im, covr_im, resl_mag, conf_mag, covr_mag

    def plot_errors(self, *args):
        row_count = len(args)

        fig, axes = plt.subplots(row_count, 3, figsize=(13, 4 * row_count))
        if row_count == 1:
            name, predictions, confidence_intervals = args[0]
            (
                resl_im,
                conf_im,
                covr_im,
                resl_mag,
                conf_mag,
                covr_mag,
            ) = self._get_images(predictions, confidence_intervals)
            self.plot_error(
                fig,
                axes,
                name,
                resl_im,
                conf_im,
                covr_im,
                resl_mag,
                conf_mag,
                covr_mag,
            )
        else:
            resl_ims = []
            conf_ims = []
            covr_ims = []
            resl_mag = 0.0
            conf_mag = 0.0
            covr_mag = 0.0
            for i, (_, predictions, confidence_intervals) in enumerate(args):
                (
                    resl_im,
                    conf_im,
                    covr_im,
                    resl_mag_,
                    conf_mag_,
                    covr_mag_,
                ) = self._get_images(predictions, confidence_intervals)
                resl_ims.append(resl_im)
                conf_ims.append(conf_im)
                covr_ims.append(covr_im)
                resl_mag = np.max([resl_mag, resl_mag_])
                conf_mag = np.max([conf_mag, conf_mag_])
                covr_mag = np.max([covr_mag, covr_mag_])
            for i, (name, _, _) in enumerate(args):
                self.plot_error(
                    fig,
                    axes[i, :],
                    name,
                    resl_ims[i],
                    conf_ims[i],
                    covr_ims[i],
                    resl_mag,
                    conf_mag,
                    covr_mag,
                )
        plt.tight_layout()
        plt.show()

    def _label_ax(self, ax, title):
        ax.set_title(title, fontsize=self._title_fsize)
        ax.set_xlabel("Axis 0", fontsize=self._axis_fsize)
        ax.set_ylabel("Axis 1", fontsize=self._axis_fsize)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_error(
        self,
        fig,
        axes,
        name,
        resl_im,
        conf_im,
        covr_im,
        resl_mag,
        conf_mag,
        covr_mag,
    ):
        self._label_ax(axes[0], f"{name} residual")
        im0 = axes[0].imshow(
            resl_im, vmin=-resl_mag, vmax=resl_mag, cmap="coolwarm"
        )
        fig.colorbar(im0, ax=axes[0])

        self._label_ax(axes[1], f"{name} CI Magnitude")
        im1 = axes[1].imshow(conf_im, vmin=0.0, vmax=conf_mag, cmap="inferno")
        fig.colorbar(im1, ax=axes[1])

        self._label_ax(axes[2], f"{name} |Residual| - CI")
        im2 = axes[2].imshow(
            covr_im, vmin=-covr_mag, vmax=covr_mag, cmap="coolwarm"
        )
        fig.colorbar(im2, ax=axes[2])


def get_length_scale(muygps):
    ls = muygps.kernel.deformation.length_scale
    if isinstance(ls, dict):
        return np.array([ls[x]() for x in ls])
    else:
        return ls()


def print_results(targets, *args, **kwargs):
    table = list()
    for arg in args:
        (
            name,
            muygps,
            means,
            variances,
            confidence_intervals,
            coverage,
        ) = arg
        table.append(
            [
                name,
                muygps.kernel.smoothness(),
                get_length_scale(muygps),
                muygps.noise(),
                muygps.scale()[0],
                np.sqrt(mse_fn(means, targets)),
                np.mean(variances),
                np.mean(confidence_intervals),
                coverage,
            ]
        )
    return pd.DataFrame(
        table,
        columns=[
            "name",
            "smoothness",
            "length scale",
            "noise variance",
            "variance scale",
            "rmse",
            "mean variance",
            "mean confidence interval",
            "coverage",
        ],
    ).style.hide(axis="index")


def print_fast_results(targets, *args, **kwargs):
    table = list()
    for arg in args:
        (
            name,
            time,
            muygps,
            means,
        ) = arg
        table.append(
            [
                name,
                np.sqrt(mse_fn(means, targets)),
                time,
                muygps.noise(),
            ]
        )
    return pd.DataFrame(
        table,
        columns=[
            "name",
            "rmse",
            "timing results",
            "noise variance",
        ],
    ).style.hide(axis="index")
