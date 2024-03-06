# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> refframe
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 10:06
=================================================='''
import numpy as np
from localization.camera import Camera
from colmap_utils.camera_intrinsics import intrinsics_from_camera
from colmap_utils.read_write_model import qvec2rotmat


class RefFrame:
    def __init__(self, camera: Camera, id: int, qvec: np.ndarray, tvec: np.ndarray,
                 point3D_ids: np.ndarray = None, keypoints: np.ndarray = None,
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
        self.descriptors = None
        self.keypoint_segs = None
        self.xyzs = None

    def get_keypoints_by_sid(self, sid: int, point3Ds: dict):
        mask = (self.keypoint_segs == sid)
        return {
            'points3D_ids': self.point3D_ids[mask],
            'keypoints': self.keypoints[mask][:, :2],
            'descriptors': self.descriptors[mask],
            'scores': self.keypoints[mask][:, 2],
            'xyzs': self.xyzs[mask],
            'camera': self.camera,
        }

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
            'points3D_ids': np.array(valid_p3d_ids),
            'keypoints': np.array(valid_kpts),
            'descriptors': np.array(valid_descs),
            'scores': np.array(valid_scores),
            'xyzs': np.array(valid_xyzs),
        }

    def get_keypoints(self, point3Ds: dict):
        return {
            'points3D_ids': self.point3D_ids,
            'keypoints': self.keypoints[:, :2],
            'descriptors': self.descriptors,
            'scores': self.keypoints[:, 2],
            'xyzs': self.xyzs,
            'camera': self.camera,
        }

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
            'points3D_ids': np.array(valid_p3d_ids),
            'keypoints': np.array(valid_kpts),
            'descriptors': np.array(valid_descs),
            'scores': 1 / np.clip(np.array(valid_scores) * 5, a_min=1., a_max=20.),
            'xyzs': np.array(valid_xyzs),
            'camera': self.camera,
        }

    def associate_keypoints_with_point3Ds(self, point3Ds: dict):
        xyzs = []
        descs = []
        scores = []
        p3d_ids = []
        kpt_sids = []
        for i, v in enumerate(self.point3D_ids):
            if v in point3Ds.keys():
                p3d = point3Ds[v]
                p3d_ids.append(v)
                xyzs.append(p3d.xyz)
                descs.append(p3d.descriptor)
                scores.append(p3d.error)

                kpt_sids.append(p3d.seg_id)

        xyzs = np.array(xyzs)
        if xyzs.shape[0] == 0:
            return False

        descs = np.array(descs)
        scores = 1 / np.clip(np.array(scores) * 5, a_min=1., a_max=20.)
        p3d_ids = np.array(p3d_ids)
        uvs = self.project(xyzs=xyzs)
        self.keypoints = np.hstack([uvs, scores.reshape(-1, 1)])
        self.descriptors = descs
        self.point3D_ids = p3d_ids
        self.xyzs = xyzs
        self.keypoint_segs = np.array(kpt_sids)

        return True

    def project(self, xyzs):
        '''
        :param xyzs: [N, 3]
        :return:
        '''
        K = intrinsics_from_camera(camera_model=self.camera.model, params=self.camera.params)  # [3, 3]
        Rcw = qvec2rotmat(self.qvec)
        tcw = self.tvec.reshape(3, 1)
        Tcw = np.eye(4, dtype=float)
        Tcw[:3, :3] = Rcw
        Tcw[:3, 3:] = tcw
        xyzs_homo = np.hstack([xyzs, np.ones(shape=(xyzs.shape[0], 1))])  # [N 4]

        xyzs_cam = Tcw @ xyzs_homo.transpose()  # [4, N]
        uvs = K @ xyzs_cam[:3, :]  # [3, N]
        uvs[:2, :] = uvs[:2, :] / uvs[2, :]
        return uvs[:2, :].transpose()
