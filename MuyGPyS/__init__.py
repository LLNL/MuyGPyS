# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Public MuyGPyS modules and functions."""

__version__ = '0.4.0'

from MuyGPyS import neighbors

from MuyGPyS.examples import classify
from MuyGPyS.examples import regress
from MuyGPyS.examples import two_class_classify_uq

from MuyGPyS.gp import distance
from MuyGPyS.gp import kernels
from MuyGPyS.gp import muygps

from MuyGPyS.optimize import batch
from MuyGPyS.optimize import chassis
from MuyGPyS.optimize import objective
