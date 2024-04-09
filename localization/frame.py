# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> frame
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   01/03/2024 10:08
=================================================='''
from collections import defaultdict

import numpy as np
import torch
import pycolmap

from localization.camera import Camera
from localization.utils import compute_pose_error


class Frame:
    def __init__(self, image: np.ndarray, camera: pycolmap.Camera, id: int, name: str = None, qvec=None, tvec=None,
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
        self.segmentations = None  # [N C]
        self.seg_scores = None  # [N C]
        self.seg_ids = None  # [N, 1]
        self.point3D_ids = None  # [N, 1]
        self.xyzs = None

        self.gt_qvec = None
        self.gt_tvec = None

        self.matched_scene_name = None
        self.matched_keypoints = None
        self.matched_keypoint_ids = None
        self.matched_xyzs = None
        self.matched_point3D_ids = None
        self.matched_inliers = None
        self.matched_sids = None
        self.matched_order = None

        self.refinement_reference_frame_ids = None
        self.image_rec = None
        self.image_matching = None
        self.image_inlier = None
        self.reference_frame_name = None
        self.image_matching_tmp = None
        self.image_inlier_tmp = None
        self.reference_frame_name_tmp = None

        self.tracking_status = None

        self.time_feat = 0
        self.time_rec = 0
        self.time_loc = 0
        self.time_ref = 0

    def update_point3ds_old(self):
        pt = torch.from_numpy(self.keypoints[:, :2]).unsqueeze(-1)  # [M 2 1]
        mpt = torch.from_numpy(self.matched_keypoints[:, :2].transpose()).unsqueeze(0)  # [1 2 N]
        dist = torch.sqrt(torch.sum((pt - mpt) ** 2, dim=1))
        values, ids = torch.topk(dist, dim=1, k=1, largest=False)
        values = values[:, 0].numpy()
        ids = ids[:, 0].numpy()
        mask = (values < 1)  # 1 pixel error
        self.point3D_ids = np.zeros(shape=(self.keypoints.shape[0],), dtype=int) - 1
        self.point3D_ids[mask] = self.matched_point3D_ids[ids[mask]]

        # self.xyzs = np.zeros(shape=(self.keypoints.shape[0], 3), dtype=float)
        inlier_mask = self.matched_inliers
        self.xyzs[mask] = self.matched_xyzs[ids[mask]]
        self.seg_ids[mask] = self.matched_sids[ids[mask]]

    def update_point3ds(self):
        # print('Frame: update_point3ds: ', self.matched_keypoint_ids.shape, self.matched_xyzs.shape,
        #       self.matched_sids.shape, self.matched_point3D_ids.shape)
        self.xyzs[self.matched_keypoint_ids] = self.matched_xyzs
        self.seg_ids[self.matched_keypoint_ids] = self.matched_sids
        self.point3D_ids[self.matched_keypoint_ids] = self.matched_point3D_ids

    def add_keypoints(self, keypoints: np.ndarray, descriptors: np.ndarray):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.initialize_localization_variables()

    def add_segmentations(self, segmentations: torch.Tensor, filtering_threshold: float):
        '''
        :param segmentations: [number_points number_labels]
        :return:
        '''
        seg_scores = torch.softmax(segmentations, dim=-1)
        if filtering_threshold > 0:
            scores_background = seg_scores[:, 0]
            non_bg_mask = (scores_background < filtering_threshold)
            print('pre filtering before: ', self.keypoints.shape)
            if torch.sum(non_bg_mask) >= 0.4 * seg_scores.shape[0]:
                self.keypoints = self.keypoints[non_bg_mask.cpu().numpy()]
                self.descriptors = self.descriptors[non_bg_mask.cpu().numpy()]
                # print('pre filtering after: ', self.keypoints.shape)

                # update localization variables
                self.initialize_localization_variables()

                segmentations = segmentations[non_bg_mask]
                seg_scores = seg_scores[non_bg_mask]
            print('pre filtering after: ', self.keypoints.shape)

        # extract initial segmentation info
        self.segmentations = segmentations.cpu().numpy()
        self.seg_scores = seg_scores.cpu().numpy()
        self.seg_ids = segmentations.max(dim=-1)[1].cpu().numpy() - 1  # should start from 0

    def filter_keypoints(self, seg_scores: np.ndarray, filtering_threshold: float):
        scores_background = seg_scores[:, 0]
        non_bg_mask = (scores_background < filtering_threshold)
        print('pre filtering before: ', self.keypoints.shape)
        if np.sum(non_bg_mask) >= 0.4 * seg_scores.shape[0]:
            self.keypoints = self.keypoints[non_bg_mask]
            self.descriptors = self.descriptors[non_bg_mask]
            print('pre filtering after: ', self.keypoints.shape)

            # update localization variables
            self.initialize_localization_variables()
            return non_bg_mask
        else:
            print('pre filtering after: ', self.keypoints.shape)
            return None

    def compute_pose_error(self, pred_qvec=None, pred_tvec=None):
        if pred_qvec is not None and pred_tvec is not None:
            if self.gt_qvec is not None and self.gt_tvec is not None:
                return compute_pose_error(pred_qcw=pred_qvec, pred_tcw=pred_tvec,
                                          gt_qcw=self.gt_qvec, gt_tcw=self.gt_tvec)
            else:
                return 100, 100

        if self.qvec is None or self.tvec is None or self.gt_qvec is None or self.gt_tvec is None:
            return 100, 100
        else:
            err_q, err_t = compute_pose_error(pred_qcw=self.qvec, pred_tcw=self.tvec,
                                              gt_qcw=self.gt_qvec, gt_tcw=self.gt_tvec)
            return err_q, err_t

    def get_intrinsics(self) -> np.ndarray:
        camera_model = self.camera.model.name
        params = self.camera.params
        if camera_model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = params[0]
            cx = params[1]
            cy = params[2]
        elif camera_model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx = params[0]
            fy = params[1]
            cx = params[2]
            cy = params[3]
        else:
            raise Exception("Camera model not supported")

        # intrinsics
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def get_dominate_seg_id(self):
        counts = np.bincount(self.seg_ids[self.seg_ids > 0])
        return np.argmax(counts)

    def clear_localization_track(self):
        self.matched_scene_name = None
        self.matched_keypoints = None
        self.matched_xyzs = None
        self.matched_point3D_ids = None
        self.matched_inliers = None
        self.matched_sids = None

        self.refinement_reference_frame_ids = None

    def initialize_localization_variables(self):
        nkpt = self.keypoints.shape[0]
        self.seg_ids = np.zeros(shape=(nkpt,), dtype=int) - 1
        self.point3D_ids = np.zeros(shape=(nkpt,), dtype=int) - 1
        self.xyzs = np.zeros(shape=(nkpt, 3), dtype=float)
