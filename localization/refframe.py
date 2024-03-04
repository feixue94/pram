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
                 points3d_ids: np.ndarray = None, keypoints: np.ndarray = None, seg_id: int = None,
                 name: str = None, scene_name: str = None):
        self.cfg = cfg
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.name = name
        self.scene_name = scene_name
        self.width = cfg['width']
        self.height = cfg['height']
        self.image_size = np.array([self.height, self.width])

        self.points3d_ids = points3d_ids
        self.keypoints = keypoints
        self.seg_id = seg_id
