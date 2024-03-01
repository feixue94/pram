# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> frame
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   01/03/2024 10:08
=================================================='''
import numpy as np
from colmap_utils.read_write_model import qvec2rotmat


class Frame:
    def __init__(self, cfg, id, name=None, image_size=None, qvec=None, tvec=None, scene_name=None,
                 reference_frame=None):
        self.cfg = cfg
        self.id = id
        self.name = name
        self.image_size = image_size
        self.qvec = qvec
        self.tvec = tvec
        self.scene_name = scene_name
        self.reference_frame = reference_frame

        self.keypoints = None  # [N, 3]
        self.descriptors = None  # [N, D]
        self.seg_ids = None  # [N, 1]
        self.points3d = None
