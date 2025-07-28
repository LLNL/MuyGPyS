#!/usr/bin/env bash

# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# usage:
# $ cd ${MUYGPYS_ROOT}
# $ sh ./scripts/flake_lint

flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
