#!/usr/bin/env python
# encoding: utf-8
"""
@file stargal.py

Created by priest2 on 2020-10-29

End-to-end application of LKGP to preprocessed star-gal dataset.
"""

import numpy as np

from muyscans.data.load import make_stargal  # $, normalize
from muyscans.examples.classify import do_classify


def do_stargal(fname="data/star-gal/galstar.csv", **kwargs):
    train, test = make_stargal(fname=fname)
    predicted_labels = do_classify(train, test, **kwargs)
    return train, test, predicted_labels
