# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> tracker
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/02/2024 16:58
=================================================='''
import numpy as np
import torch
import pycolmap
from localization.simglelocmap import SingleLocMap
from localization.multilocmap import MultiLocMap
from localization.frame import Frame


class Tracker:
    def __init__(self, locMap, viewer, matcher, loc_config):
        self.locMap = locMap
        self.viewer = viewer
        self.matcher = matcher
        self.loc_config = loc_config

        self.lost = True

        self.curr_frame = None
        self.last_frame = None

    def track(self, frame: Frame):
        self.curr_frame = frame
        reference_frame = self.last_frame.reference_frame

        mids1, last_mids = self.match_frame(frame=self.curr_frame, reference_frame=self.last_frame)

        matched_kpts_last = frame.keypoints[mids1, :2]  # [N 2]
        matched_p3ds_last = self.last_frame.points3d[last_mids]  # [N 3]

        ret = pycolmap.absolute_pose_estimation(matched_kpts_last, matched_p3ds_last, frame.cfg,
                                                max_error_px=self.config['localization']['threshold'])

        track_reference = False
        success = ret['success']
        inliers = np.array(ret['inliers'])
        if success:
            num_inliers = ret['num_inliers']
            if num_inliers < self.loc_config['tracking_inliers']:
                track_reference = True

        if not track_reference:
            pass

        # tracking reference frame with graph-matcher
        mids2, ref_mids = self.match_frame(frame=self.curr_frame, reference_frame=reference_frame)
        matched_kpts_ref = frame.keypoints[mids1, :2]  # [N 2]
        matched_p3ds_ref = reference_frame.points3d[last_mids]  # [N 3]

        ret = pycolmap.absolute_pose_estimation(np.vstack([matched_kpts_last[inliers], matched_kpts_ref]),
                                                np.vstack([matched_p3ds_last[inliers], matched_p3ds_ref]),
                                                max_error_px=self.config['localization']['threshold'])

        success = ret['success']
        inliers = np.array(ret['inliers'])
        if success:
            num_inliers = ret['num_inliers']
            if num_inliers < self.loc_config['tracking_inliers']:
                do_refinement = True

    def refine_pose(self):
        pass

    def update_current_frame(self):
        pass

    @torch.no_grad()
    def match_frame(self, frame: Frame, reference_frame: Frame):
        matches = self.matcher({
            'descriptors0': torch.from_numpy(frame.descriptors)[None].cuda().float(),
            'keypoints0': torch.from_numpy(frame.keypoints[:, :2])[None].cuda().float(),
            'scores0': torch.from_numpy(frame.keypoints[:, 2:])[None].cuda().float(),
            'image_shape0': (1, 3, frame.image_size[0], frame.image_size[1]),

            'descriptors1': torch.from_numpy(reference_frame.descriptors)[None].cuda().float(),
            'keypoints1': torch.from_numpy(reference_frame.keypoints[:, :2])[None].cuda().float(),
            'scores1': torch.from_numpy(reference_frame.keypoints[:, 2:])[None].cuda().float(),
            'image_shape1': (1, 3, reference_frame.image_size[0], reference_frame.image_size[1]),

        })['matches0'][0].cpu().numpy()

        ids1 = np.arange(matches.shape[0])
        ids2 = matches
        ids1 = ids1[matches >= 0]
        ids2 = ids2[matches >= 0]

        return ids1, ids2
