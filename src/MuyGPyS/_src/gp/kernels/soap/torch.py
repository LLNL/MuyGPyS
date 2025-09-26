# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


def _omega(diffs, sensitivity):
    raise NotImplementedError("MPI backend not yet supported for SOAPKernel")


def _T1(diffs, sensitivity):
    raise NotImplementedError("MPI backend not yet supported for SOAPKernel")


def _T2(diffs, sensitivity):
    raise NotImplementedError("MPI backend not yet supported for SOAPKernel")


def _T3(diffs, sensitivity):
    raise NotImplementedError("MPI backend not yet supported for SOAPKernel")


def _Knm(omega, T1, T2, T3, sensitivity):
    raise NotImplementedError("MPI backend not yet supported for SOAPKernel")


def _soap_fn(diffs, sensitivity):
    raise NotImplementedError("MPI backend not yet supported for SOAPKernel")
