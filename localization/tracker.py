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
from localization.simglelocmap import SingleLocMap
from localization.base_model import dynamic_load
import localization.matchers as matchers
from localization.match_features import confs as matcher_confs


class Tracker:
    def __init__(self, locMap, matcher, config):
        self.locMap = locMap
        self.matcher = matcher
        self.config = config

        self.lost = True

        self.curr_frame = None
        self.last_frame = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = dynamic_load(matchers, 'NNM')
        self.nn_matcher = Model(matcher_confs['NNM']['model']).eval().to(device)

    def run(self, frame: Frame):
        print('Start tracking...')
        self.last_frame = self.curr_frame
        self.curr_frame = frame

        reference_frame = self.last_frame.reference_frame

        mids1, last_mids = self.match_frame(frame=self.curr_frame, reference_frame=self.last_frame)

        matched_kpts_last = frame.keypoints[mids1, :2]  # [N 2]
        matched_p3ds_last = self.last_frame.points3d[last_mids]  # [N 3]
        print('Tracking: ', matched_kpts_last.shape, matched_p3ds_last.shape)

        ret = pycolmap.absolute_pose_estimation(matched_kpts_last, matched_p3ds_last, frame.cfg,
                                                max_error_px=self.config['localization']['threshold'])

        track_reference = True
        success = ret['success']
        if success:
            inliers = np.array(ret['inliers'])
            num_inliers = ret['num_inliers']
            print('Tracking with last frame with {:d}/{:d} inliers'.format(num_inliers, matched_p3ds_last.shape[0]))
            if num_inliers > self.config['localization']['tracking_inliers']:
                track_reference = False
                self.lost = False

        if not track_reference:
            self.update_current_frame(mids=mids1[inliers],
                                      mp3ds=matched_p3ds_last[inliers],
                                      qvec=ret['qvec'], tvec=ret['tvec'],
                                      reference_frame=reference_frame)
            return
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
            if num_inliers > self.config['localization']['tracking_inliers']:
                do_refinement = False
                self.lost = False
                self.update_current_frame(mids=mids[inliers], mp3ds=matched_p3ds[inliers], qvec=ret['qvec'],
                                          tvec=ret['tvec'], reference_frame=reference_frame)

        if do_refinement:
            pass

    def track_last_frame(self, curr_frame: Frame, last_frame: Frame):
        curr_kpts = curr_frame.keypoints
        curr_descs = curr_frame.descriptors
        last_mask = last_frame.point3D_ids >= 0

        last_kpts = last_frame.keypoints[last_mask]
        last_descs = last_frame.descriptors[last_mask]
        last_xyzs = last_frame.xyzs[last_mask]
        last_point3D_ids = last_frame.point3D_ids[last_mask]

        indices = self.nn_matcher({
            'descriptors0': torch.from_numpy(curr_descs.transpose()).float().cuda()[None],
            'descriptors1': torch.from_numpy(last_descs.transpose()).float().cuda()[None],
        })[0].cpu().numpy()

        valid = indices >= 0

        matched_kpts = curr_kpts[valid]
        matched_xyzs = last_xyzs[indices[valid]]
        matched_point3D_ids = last_point3D_ids[valid]

        ret = pycolmap.absolute_pose_estimation(matched_kpts + 0.5, matched_xyzs,
                                                curr_frame.camera._asdict(),
                                                max_error_px=self.config['localization']['threshold'])

        success = ret['success']
        inliers = np.array(ret['inliers'])
        if success:
            num_inliers = ret['num_inliers']
            if num_inliers > self.config['localization']['tracking_inliers']:
                self.lost = False

                curr_frame.matched_keypoints = matched_kpts[inliers]
                curr_frame.matched_xyzs = matched_xyzs[inliers]
                curr_frame.reference_frame_id = last_frame.reference_frame_id
                curr_frame.matched_points3D_ids = matched_point3D_ids[inliers]
                return True

        self.lost = True
        return False

    def update_current_frame(self, mids, mp3ds, qvec, tvec, reference_frame):
        self.curr_frame.points3d[mids] = mp3ds
        self.curr_frame.points3d_mask[mids] = True
        self.curr_frame.qvec = qvec
        self.curr_frame.tvec = tvec
        self.curr_frame.reference_frame = reference_frame
        self.lost = False

    def update_current_frame_from_reloc(self, frame, mp2ds, mp3ds, qvec, tvec, reference_frame):
        frame.qvec = qvec
        frame.tvec = tvec
        frame.reference_frame = reference_frame

        kpts = frame.keypoints[:, :2]
        dist = torch.from_numpy(mp2ds)[..., None] - torch.from_numpy(kpts).transpose(0, 1)[None]
        dist = torch.sum(dist ** 2, dim=1)  # [M, N]
        l2dists, ids = torch.topk(dist, k=1, largest=False, dim=1)
        ids = ids.cpu().numpy().reshape(-1, )
        mask = np.zeros(shape=(frame.keypoints.shape[0],), dtype=bool)
        mask[ids] = True
        frame.points3d_mask = mask
        frame.points3d[ids] = mp3ds

        self.curr_frame = frame

        self.lost = False

    @torch.no_grad()
    def match_frame(self, frame: Frame, reference_frame: Frame):
        print('match: ', frame.keypoints.shape, reference_frame.keypoints.shape)
        matches = self.matcher({
            'descriptors0': torch.from_numpy(frame.descriptors)[None].cuda().float(),
            'keypoints0': torch.from_numpy(frame.keypoints[:, :2])[None].cuda().float(),
            'scores0': torch.from_numpy(frame.keypoints[:, 2])[None].cuda().float(),
            'image_shape0': (1, 3, frame.image_size[0], frame.image_size[1]),

            # 'descriptors0': torch.from_numpy(reference_frame.descriptors)[None].cuda().float(),
            # 'keypoints0': torch.from_numpy(reference_frame.keypoints[:, :2])[None].cuda().float(),
            # 'scores0': torch.from_numpy(reference_frame.keypoints[:, 2])[None].cuda().float(),
            # 'image_shape0': (1, 3, reference_frame.image_size[0], reference_frame.image_size[1]),

            'descriptors1': torch.from_numpy(reference_frame.descriptors)[None].cuda().float(),
            'keypoints1': torch.from_numpy(reference_frame.keypoints[:, :2])[None].cuda().float(),
            'scores1': torch.from_numpy(reference_frame.keypoints[:, 2])[None].cuda().float(),
            'image_shape1': (1, 3, reference_frame.image_size[0], reference_frame.image_size[1]),

        })['matches0'][0].cpu().numpy()

        print('valid: ', np.sum(matches >= 0))

        ids1 = np.arange(matches.shape[0])
        ids2 = matches
        ids1 = ids1[matches >= 0]
        ids2 = ids2[matches >= 0]

        mask_p3ds = reference_frame.points3d_mask[ids2]
        ids1 = ids1[mask_p3ds]
        ids2 = ids2[mask_p3ds]

        return ids1, ids2
