#!/usr/bin/env bash

# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/tce/packages/cuda/cuda-11.4.1/lib64:/collab/usr/global/tools/nvidia/cudnn/toss_3_x86_64_ib/cudnn-8.1.1/lib64:/opt/cudatoolkit/11.4/lib64
export PATH=${PATH}:/usr/tce/packages/cuda/cuda-11.4.1/bin