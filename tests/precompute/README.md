# Testing bed for fast precomputation for mean prediction in MuyGPs

This directory contains tests meant specifically for accelerated mean estimation
using precomputation and MuyGPs as described in 
[Fast Gaussian Process Posterior Mean Prediction via Local Cross Validation and Precomputation](https://arxiv.org/abs/2205.10879v1). This feature currently does not support MPI, and can only be run in
JAX, Torch, and Numpy backends.