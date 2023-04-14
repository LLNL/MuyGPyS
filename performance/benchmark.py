# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import argparse
import h5py
import pickle

from time import perf_counter_ns

import MuyGPyS._src.math as mm
from MuyGPyS import config
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.distortion import IsotropicDistortion
from MuyGPyS.gp.kernels import Hyperparameter, RBF, Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp.tensors import make_train_tensors, make_predict_tensors
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.loss import lool_fn, mse_fn, cross_entropy_fn
from MuyGPyS.optimize.objective import make_obj_fn
from MuyGPyS.optimize.sigma_sq import (
    make_sigma_sq_optim,
    muygps_analytic_sigma_sq_optim,
)


def print_line():
    print(
        "=================================================="
        "=================================================="
    )


def extract_data(heaton_file):
    with h5py.File(heaton_file, "r") as f:
        train_features = mm.array(f["train"]["features"])
        train_responses = mm.array(f["train"]["responses"])
        train_mask = mm.array(f["train"]["mask"])
        test_features = mm.array(f["test"]["features"])
        test_responses = mm.array(f["test"]["responses"])
        test_mask = mm.array(f["test"]["mask"])
        domain_shape = mm.array(f["shape"])
    return (
        train_features,
        train_responses,
        train_mask,
        test_features,
        test_responses,
        test_mask,
        domain_shape,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark major functions of MuyGPyS"
    )
    parser.add_argument("file", type=str, help="hdf5 data file.")
    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        default=None,
        help="pickle archive file for results.",
    )
    parser.add_argument(
        "-b",
        "--batch-count",
        type=int,
        default=500,
        help="number of batch elements to sample.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1,
        help="number of timing iterations to run.",
    )
    parser.add_argument(
        "-k",
        "--nn-count",
        type=int,
        default=40,
        help="number of nearest neighbors to query.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="if set, print verbose messages.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
    muygps = MuyGPS(
        Matern(
            metric=IsotropicDistortion(
                "l2", length_scale=Hyperparameter(1.0, (1e-1, 1e1))
            ),
            nu=Hyperparameter(1 / 2),
        ),
        eps=HomoscedasticNoise(1e-3),
    )

    pipeline = BenchmarkPipeline(muygps, args, nn_kwargs)

    pipeline()
    pipeline.serialize()


def get_obj_fn(
    loss_method,
    loss_fn,
    muygps,
    pairwise_diffs,
    crosswise_diffs,
    batch_nn_targets,
    batch_targets,
):
    kernel_fn = muygps.kernel.get_opt_fn()
    mean_fn = muygps.get_opt_mean_fn()
    var_fn = muygps.get_opt_var_fn()
    sigma_sq_fn = make_sigma_sq_optim("analytic", muygps)
    return make_obj_fn(
        "loo_crossval",
        loss_method,
        loss_fn,
        kernel_fn,
        mean_fn,
        var_fn,
        sigma_sq_fn,
        pairwise_diffs,
        crosswise_diffs,
        batch_nn_targets,
        batch_targets,
    )


class BenchmarkPipeline:
    def __init__(self, muygps, params, nn_kwargs):
        self._muygps = muygps
        self._params = params
        self.prepare_data(nn_kwargs)
        self.values = dict()
        self.timings = dict()
        self.distortion_fn = benchmark_fn(
            self._muygps.kernel._distortion_fn, self._params
        )
        self.kernel_only_fn = benchmark_fn(
            self._muygps.kernel._kernel_fn, self._params
        )
        self.kernel_fn = benchmark_fn(self._muygps.kernel, self._params)
        self.mean_fn = benchmark_fn(self._muygps.posterior_mean, self._params)
        self.var_fn = benchmark_fn(
            self._muygps.posterior_variance, self._params
        )
        self.mse_fn = benchmark_fn(mse_fn, self._params)
        self.mse_obj_fn = benchmark_fn(
            get_obj_fn(
                "mse",
                mse_fn,
                self._muygps,
                self._batch_pairwise_diffs,
                self._batch_crosswise_diffs,
                self._batch_nn_targets,
                self._batch_targets,
            ),
            self._params,
        )
        self.lool_fn = benchmark_fn(lool_fn, self._params)
        self.lool_obj_fn = benchmark_fn(
            get_obj_fn(
                "lool",
                lool_fn,
                self._muygps,
                self._batch_pairwise_diffs,
                self._batch_crosswise_diffs,
                self._batch_nn_targets,
                self._batch_targets,
            ),
            self._params,
        )
        # self.cross_entropy_fn = benchmark_fn(cross_entropy_fn, params)
        self.sigma_sq_fn = benchmark_fn(
            muygps_analytic_sigma_sq_optim, self._params
        )

    def profile(self, name, value_timing):
        value, timing = value_timing
        self.values[name] = value
        self.timings[name] = timing
        if self._params.verbose is True:
            print(f"{name} : {self.timings[name]}s")
        return value

    def __call__(self):
        print(f"Begin pipeline for {self._params.iterations} iterations")
        print_line()
        # batch profiling
        batch_pairwise_dists = self.profile(
            "batch pairwise distances",
            self.distortion_fn(
                self._batch_pairwise_diffs,
                self._muygps.kernel._distortion_fn.length_scale(),
            ),
        )
        self.profile(
            "batch covariance (kernel only)",
            self.kernel_only_fn(batch_pairwise_dists),
        )
        batch_K = self.profile(
            "batch covariance", self.kernel_fn(self._batch_pairwise_diffs)
        )
        batch_Kcross = self.profile(
            "batch crosscovariance", self.kernel_fn(self._batch_crosswise_diffs)
        )
        batch_mean = self.profile(
            "batch posterior mean",
            self.mean_fn(batch_K, batch_Kcross, self._batch_nn_targets),
        )
        batch_var = self.profile(
            "batch posterior variance", self.var_fn(batch_K, batch_Kcross)
        )
        # opt profiling
        self.profile("mse fn", self.mse_fn(batch_mean, self._batch_targets))
        self.profile(
            "sigma_sq optim",
            self.sigma_sq_fn(
                self._muygps, self._batch_pairwise_diffs, self._batch_nn_targets
            ),
        )
        self.profile(
            "lool fn",
            self.lool_fn(
                batch_mean,
                self._batch_targets,
                batch_var,
                self._muygps.sigma_sq(),
            ),
        )
        self.profile("mse objective fn", self.mse_obj_fn(length_scale=2.0))
        self.profile("lool objective fn", self.lool_obj_fn(length_scale=2.0))
        # test profiling
        test_pairwise_dists = self.profile(
            "test pairwise distances",
            self.distortion_fn(
                self._test_pairwise_diffs,
                self._muygps.kernel._distortion_fn.length_scale(),
            ),
        )
        self.profile(
            "test covariance (kernel only)",
            self.kernel_only_fn(test_pairwise_dists),
        )
        test_K = self.profile(
            "test covariance", self.kernel_fn(self._test_pairwise_diffs)
        )

        test_Kcross = self.profile(
            "test crosscovariance", self.kernel_fn(self._test_crosswise_diffs)
        )
        self.profile(
            "test posterior mean",
            self.mean_fn(test_K, test_Kcross, self._test_nn_targets),
        )
        self.profile(
            "test posterior variance", self.var_fn(test_K, test_Kcross)
        )

    def serialize(self):
        if self._params.out_file is not None:
            with open(self._params.out_file, "wb") as f:
                print("gets here")
                pickle.dump(self.timings, f)

    def prepare_data(self, nn_kwargs):
        (
            train_features,
            train_responses,
            train_mask,
            test_features,
            test_responses,
            test_mask,
            domain_shape,
        ) = extract_data(self._params.file)

        train_count, feature_count = train_features.shape
        test_count, response_count = test_responses.shape

        nbrs_lookup = NN_Wrapper(
            train_features, self._params.nn_count, **nn_kwargs
        )

        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, self._params.batch_count, train_count
        )

        (
            self._batch_crosswise_diffs,
            self._batch_pairwise_diffs,
            self._batch_targets,
            self._batch_nn_targets,
        ) = make_train_tensors(
            batch_indices,
            batch_nn_indices,
            train_features,
            train_responses,
        )

        test_nn_indices, _ = nbrs_lookup.get_nns(test_features)

        (
            self._test_crosswise_diffs,
            self._test_pairwise_diffs,
            self._test_nn_targets,
        ) = make_predict_tensors(
            mm.arange(test_count),
            test_nn_indices,
            test_features,
            train_features,
            train_responses,
        )


def print_timing(rank, name, mode, timing, verbose):
    if rank == 0:
        if verbose is True:
            print(f"{name} {mode} runtime {timing}s")
        else:
            print(timing)


def benchmark_fn(fn, params):
    def profiler_fn(*args, **kwargs):
        if config.state.backend == "jax":
            fn(*args, **kwargs)
        value = None
        timing = 0.0
        for _ in range(params.iterations):
            start_time = perf_counter_ns()
            value = fn(*args, **kwargs)
            end_time = perf_counter_ns()
            timing += (end_time - start_time) / 1e9
        return value, timing

    return profiler_fn


if __name__ == "__main__":
    main()
