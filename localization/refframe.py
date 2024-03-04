# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> refframe
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 10:06
=================================================='''
import numpy as np


class RefFrame:
    def __init__(self, cfg: dict, id: int, qvec: np.ndarray, tvec: np.ndarray,
                 points3d_ids: np.ndarray = None,
                 name: str = None, scene_name: str = None):
        self.cfg = cfg
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.name = name
        self.scene_name = scene_name
        self.image_size = np.array([cfg['height'], cfg['width']])

        self.points3d_ids = points3d_ids
        self.keypoints = None
        self.descriptors = None
