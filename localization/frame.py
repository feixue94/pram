# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> frame
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   01/03/2024 10:08
=================================================='''
import numpy as np
import torch

from localization.camera import Camera
from localization.utils import compute_pose_error


class Frame:
    def __init__(self, image: np.ndarray, camera: Camera, id: int, name: str = None, qvec=None, tvec=None,
                 scene_name=None,
                 reference_frame_id=None):
        self.image = image
        self.camera = camera
        self.id = id
        self.name = name
        self.image_size = np.array([camera.height, camera.width])
        self.qvec = qvec
        self.tvec = tvec
        self.scene_name = scene_name
        self.reference_frame_id = reference_frame_id

        self.keypoints = None  # [N, 3]
        self.descriptors = None  # [N, D]
        self.seg_ids = None  # [N, 1]
        self.points3d = None
        self.segmentations = None

        self.matched_scene_name = None
        self.matched_keypoints = None
        self.matched_xyzs = None
        self.matched_points3D_ids = None
        self.matched_inliers = None
        self.matched_sids = None
        self.gt_qvec = None
        self.gt_tvec = None
        self.tracking_status = None
        self.refinement_reference_frame_ids = None
        self.image_rec = None
        self.image_matching = None
        self.image_inlier = None

        self.time_feat = 0
        self.time_rec = 0
        self.time_loc = 0
        self.time_ref = 0

    def update_features(self, keypoints, descriptors, points3d=None, seg_ids=None):
        self.keypoints = keypoints
        self.descriptors = descriptors
        n = keypoints.shape[0]
        if points3d is None:
            self.points3d = np.zeros(shape=(n, 3), dtype=float)
            self.points3d_mask = np.zeros(shape=(n,), dtype=bool)

        if seg_ids is None:
            self.seg_ids = np.zeros(shape=(n,), dtype=int) - 1

    def filter_keypoints(self, seg_scores: np.ndarray, filtering_threshold: float):
        scores_background = seg_scores[:, 0]
        non_bg_mask = (scores_background < filtering_threshold)
        print('pre filtering before: ', self.keypoints.shape)
        if np.sum(non_bg_mask) >= 0.4 * seg_scores.shape[0]:
            self.keypoints = self.keypoints[non_bg_mask]
            self.descriptors = self.descriptors[non_bg_mask]
            print('pre filtering after: ', self.keypoints.shape)
            return non_bg_mask
        else:
            print('pre filtering after: ', self.keypoints.shape)
            return None

    def compute_pose_error(self):
        if self.qvec is None or self.tvec is None or self.gt_qvec is None or self.gt_tvec is None:
            return 100, 100
        else:
            err_q, err_t = compute_pose_error(pred_qcw=self.qvec, pred_tcw=self.tvec,
                                              gt_qcw=self.gt_qvec, gt_tcw=self.gt_tvec)
            return err_q, err_t
