# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> utils
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:31
=================================================='''
import numpy as np
import math as m


def normalize_size(x, size, scale=0.7):
    size = size.reshape([1, 2])
    norm_fac = size.max() + 0.5
    return (x - size / 2) / (norm_fac * scale)


