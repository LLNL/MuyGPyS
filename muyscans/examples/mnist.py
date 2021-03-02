#!/usr/bin/env python
# encoding: utf-8
"""
@file mnist.py

Created by priest2 on 2020-10-19

End-to-end application of LKGP to mnist.
"""

import numpy as np

from muyscans.data.load import make_mnist  # $, normalize
from muyscans.examples.classify import do_classify


def do_mnist(**kwargs):
    train, test = make_mnist()
    predicted_labels = do_classify(train, test, **kwargs)
    return train, test, predicted_labels
