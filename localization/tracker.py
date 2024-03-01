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
from localization.frame import Frame


class Tracker:
    def __init__(self, locMap, matcher, loc_config):
        self.locMap = locMap
        self.matcher = matcher
        self.loc_config = loc_config

        self.lost = True

        self.curr_frame = None
        self.last_frame = None

    def run(self, frame: Frame):
        self.last_frame = self.curr_frame
        self.curr_frame = frame

        reference_frame = self.last_frame.reference_frame

        mids1, last_mids = self.match_frame(frame=self.curr_frame, reference_frame=self.last_frame)

        matched_kpts_last = frame.keypoints[mids1, :2]  # [N 2]
        matched_p3ds_last = self.last_frame.points3d[last_mids]  # [N 3]

        ret = pycolmap.absolute_pose_estimation(matched_kpts_last, matched_p3ds_last, frame.cfg,
                                                max_error_px=self.config['localization']['threshold'])

        track_reference = True
        success = ret['success']
        inliers = np.array(ret['inliers'])
        if success:
            num_inliers = ret['num_inliers']
            if num_inliers > self.loc_config['tracking_inliers']:
                track_reference = False
                self.lost = False

        if not track_reference:
            self.update_current_frame(mids=mids1[inliers], mp3ds=matched_p3ds_last[inliers],
                                      qvec=ret['qvec'], tvec=ret['tvec'],
                                      reference_frame=reference_frame)

        # tracking reference frame with graph-matcher
        mids2, ref_mids = self.match_frame(frame=self.curr_frame, reference_frame=reference_frame)
        matched_kpts_ref = frame.keypoints[mids1, :2]  # [N 2]
        matched_p3ds_ref = reference_frame.points3d[last_mids]  # [N 3]

        mids = np.vstack([mids1[inliers], mids2])
        matched_kpts = np.vstack([matched_kpts_last[inliers], matched_kpts_ref])
        matched_p3ds = np.vstack([[matched_p3ds_last[inliers], matched_p3ds_ref]])

        ret = pycolmap.absolute_pose_estimation(matched_kpts, matched_p3ds, self.curr_frame.cfg,
                                                max_error_px=self.config['localization']['threshold'])

        do_refinement = True
        success = ret['success']
        inliers = np.array(ret['inliers'])
        if success:
            num_inliers = ret['num_inliers']
            if num_inliers > self.loc_config['tracking_inliers']:
                do_refinement = False
                self.lost = False
                self.update_current_frame(mids=mids[inliers], mp3ds=matched_p3ds[inliers], qvec=ret['qvec'],
                                          tvec=ret['tvec'], reference_frame=reference_frame)

        if do_refinement:
            pass

    def update_current_frame(self, mids, mp3ds, qvec, tvec, reference_frame):
        self.curr_frame.points3d[mids] = mp3ds
        self.curr_frame.qvec = qvec
        self.curr_frame.tvec = tvec
        self.curr_frame.reference_frame = reference_frame
        self.lost = False

    def update_current_frame_from_reloc(self, frame, mp2ds, mp3ds, qvec, tvec, reference_frame):
        
        self.lost = False

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
