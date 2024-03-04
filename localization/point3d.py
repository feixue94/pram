# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> point3d
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 10:13
=================================================='''
import numpy as np


class Point3D:
    def __init__(self, id: int, xyz: np.ndarray, error: float, refframe_id: int, seg_id: int = None,
                 descriptor: np.ndarray = None):
        self.id = id
        self.xyz = xyz
        self.error = error
        self.seg_id = seg_id
        self.refframe_id = refframe_id
        self.descriptor = descriptor
