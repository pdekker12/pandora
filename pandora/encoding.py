#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

from sklearn.preprocessing import LabelEncoder


class PandoraLabelEncoder(LabelEncoder):
    """
    Wrapper around scikit's LabelEncoder to
    enable saving and loading of a fitted
    encoder.
    """

    def __init__(self):
        super(PandoraLabelEncoder, self).__init__()

    def save(self, p):
        with open(p, 'w') as f:
            f.write('\n'.join(self.classes_))

    def load(self, p):
        self.classes_ = np.array([l.strip() for l in open(p, 'r')])
