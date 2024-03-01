# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> frame
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   01/03/2024 10:08
=================================================='''
import numpy as np


class Frame:
    def __init__(self, cfg, id, name=None, qvec=None, tvec=None, scene_name=None,
                 reference_frame=None):
        self.cfg = cfg
        self.id = id
        self.name = name
        self.image_size = np.array([cfg['height'], cfg['width']])
        self.qvec = qvec
        self.tvec = tvec
        self.scene_name = scene_name
        self.reference_frame = reference_frame

        self.keypoints = None  # [N, 3]
        self.descriptors = None  # [N, D]
        self.seg_ids = None  # [N, 1]
        self.points3d = None
        self.points3d_mask = None

    def update_features(self, keypoints, descriptors, points3d=None, seg_ids=None):
        self.keypoints = keypoints
        self.descriptors = descriptors
        n = keypoints.shape[0]
        if points3d is None:
            self.points3d = np.zeros(shape=(n, 3), dtype=float)
            self.points3d_mask = np.zeros(shape=(n,), dtype=bool)

        if seg_ids is None:
            self.seg_ids = np.zeros(shape=(n,), dtype=int) - 1

    def update_mp3ds(self, mp2ds, mp3ds):
        pass
