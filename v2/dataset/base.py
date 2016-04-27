

import os
import glob

import numpy as np
import scipy as sp

import utils


class DataSet(object):

    def next(self, **params):
        raise NotImplementedError

    @property
    def batch_size(self):
        raise NotImplementedError


