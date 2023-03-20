# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


class NullDistortion:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("NullDistortion cannot be called!")
