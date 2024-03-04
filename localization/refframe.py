# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> refframe
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 10:06
=================================================='''
import numpy as np
from localization.camera import Camera
from localization.singlemap3d import SingleMap3D


class RefFrame:
    def __init__(self, camera: Camera, id: int, qvec: np.ndarray, tvec: np.ndarray,
                 point3D_ids: np.ndarray = None, keypoints: np.ndarray = None, seg_id: int = None,
                 name: str = None, scene_name: str = None):
        self.camera = camera
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.name = name
        self.scene_name = scene_name
        self.width = camera.width
        self.height = camera.height
        self.image_size = np.array([self.height, self.width])

        self.point3D_ids = point3D_ids
        self.keypoints = keypoints
        self.seg_id = seg_id

    def get_keypoints_by_sid(self, sid: int, point3Ds: dict):
        valid_p3d_ids = []
        valid_kpts = []
        valid_descs = []
        valid_scores = []
        valid_xyzs = []
        for i, v in enumerate(self.point3D_ids):
            if v in point3Ds.keys():
                p3d = point3Ds[v]
                if p3d.seg_id == sid:
                    valid_kpts.append(self.keypoints[i])
                    valid_p3d_ids.append(v)
                    valid_xyzs.append(p3d.xyz)
                    valid_descs.append(p3d.descriptor)
                    valid_scores.append(p3d.error)
        return {
            'p3d_ids': np.array(valid_p3d_ids),
            'keypoints': np.array(valid_kpts),
            'descriptors': np.array(valid_descs),
            'scores': np.array(valid_scores),
            'xyzs': np.array(valid_xyzs),
        }

    def get_keypoints(self, point3Ds: dict):
        valid_p3d_ids = []
        valid_kpts = []
        valid_descs = []
        valid_scores = []
        valid_xyzs = []
        for i, v in enumerate(self.point3D_ids):
            if v in point3Ds.keys():
                p3d = point3Ds[v]
                valid_kpts.append(self.keypoints[i])
                valid_p3d_ids.append(v)
                valid_xyzs.append(p3d.xyz)
                valid_descs.append(p3d.descriptor)
                valid_scores.append(p3d.error)
        return {
            'p3d_ids': np.array(valid_p3d_ids),
            'keypoints': np.array(valid_kpts),
            'descriptors': np.array(valid_descs),
            'scores': 1 / np.clip(np.array(valid_scores) * 5, a_min=1., a_max=20.),
            'xyzs': np.array(valid_xyzs),
        }
