# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> tracker
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/02/2024 16:58
=================================================='''
import time
import cv2
import numpy as np
import torch
import pycolmap
from localization.frame import Frame
from localization.base_model import dynamic_load
import localization.matchers as matchers
from localization.match_features import confs as matcher_confs
from recognition.vis_seg import vis_seg_point, generate_color_dic, vis_inlier, plot_matches
from tools.common import resize_img


class Tracker:
    def __init__(self, locMap, matcher, config):
        self.locMap = locMap
        self.matcher = matcher
        self.config = config
        self.loc_config = config['localization']

        self.lost = True

        self.curr_frame = None
        self.last_frame = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = dynamic_load(matchers, 'nearest_neighbor')
        self.nn_matcher = Model(matcher_confs['NNM']['model']).eval().to(device)

    def run(self, frame: Frame):
        print('Start tracking...')
        show = self.config['localization']['show']
        self.curr_frame = frame
        ref_img = self.last_frame.image
        curr_img = self.curr_frame.image
        q_kpts = frame.keypoints

        t_start = time.time()
        ret = self.track_last_frame(curr_frame=self.curr_frame, last_frame=self.last_frame)
        self.curr_frame.time_loc = self.curr_frame.time_loc + time.time() - t_start

        if show:
            curr_matched_kpts = ret['matched_keypoints']
            ref_matched_kpts = ret['matched_ref_keypoints']
            img_loc_matching = plot_matches(img1=curr_img, img2=ref_img,
                                            pts1=curr_matched_kpts,
                                            pts2=ref_matched_kpts,
                                            inliers=np.array([True for i in range(curr_matched_kpts.shape[0])]),
                                            radius=9, line_thickness=3)
            self.curr_frame.image_matching = img_loc_matching

            q_ref_img_matching = resize_img(img_loc_matching, nh=512)

        if not ret['success']:
            show_text = 'Tracking FAILED!'
            img_inlier = vis_inlier(img=curr_img, kpts=curr_matched_kpts,
                                    inliers=[False for i in range(curr_matched_kpts.shape[0])], radius=9 + 2,
                                    thickness=2)
            q_img_inlier = cv2.putText(img=img_inlier, text=show_text, org=(30, 30),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                       thickness=2, lineType=cv2.LINE_AA)

            q_img_loc = np.hstack([resize_img(q_ref_img_matching, nh=512), resize_img(q_img_inlier, nh=512)])

            cv2.imshow('loc', q_img_loc)
            key = cv2.waitKey(self.loc_config['show_time'])
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
            return False

        ret['matched_scene_name'] = self.last_frame.scene_name
        success = self.verify_and_update(q_frame=self.curr_frame, ret=ret)

        if not success:
            return False

        if ret['num_inliers'] < 256:
            # refinement is necessary for tracking last frame
            t_start = time.time()
            ret = self.locMap.sub_maps[self.last_frame.matched_scene_name].refine_pose(self.curr_frame,
                                                                                       refinement_method=
                                                                                       self.loc_config[
                                                                                           'refinement_method'])
            self.curr_frame.time_ref = self.curr_frame.time_ref + time.time() - t_start
            ret['matched_scene_name'] = self.last_frame.scene_name
            success = self.verify_and_update(q_frame=self.curr_frame, ret=ret)

        if show:
            q_err, t_err = self.curr_frame.compute_pose_error()
            num_matches = ret['matched_keypoints'].shape[0]
            num_inliers = ret['num_inliers']
            show_text = 'Tracking, k/m/i: {:d}/{:d}/{:d}'.format(q_kpts.shape[0], num_matches, num_inliers)
            q_img_inlier = vis_inlier(img=curr_img, kpts=ret['matched_keypoints'], inliers=ret['inliers'],
                                      radius=9 + 2, thickness=2)
            q_img_inlier = cv2.putText(img=q_img_inlier, text=show_text, org=(30, 30),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                       thickness=2, lineType=cv2.LINE_AA)
            show_text = 'r_err:{:.2f}, t_err:{:.2f}'.format(q_err, t_err)
            q_img_inlier = cv2.putText(img=q_img_inlier, text=show_text, org=(30, 80),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                       thickness=2, lineType=cv2.LINE_AA)
            self.curr_frame.image_inlier = q_img_inlier

            q_img_loc = np.hstack([resize_img(q_ref_img_matching, nh=512), resize_img(q_img_inlier, nh=512)])

            cv2.imshow('loc', q_img_loc)
            key = cv2.waitKey(self.loc_config['show_time'])
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)

        self.lost = success
        return success

    def verify_and_update(self, q_frame: Frame, ret: dict):
        num_matches = ret['matched_keypoints'].shape[0]
        num_inliers = ret['num_inliers']

        q_frame.qvec = ret['qvec']
        q_frame.tvec = ret['tvec']

        q_err, t_err = q_frame.compute_pose_error()

        if num_inliers < self.loc_config['min_inliers']:
            print_text = 'Failed due to insufficient {:d} inliers,  q_err: {:.2f}, t_err: {:.2f}'.format(
                ret['num_inliers'], q_err, t_err)
            print(print_text)
            q_frame.tracking_status = False
            q_frame.clear_localization_track()
            return False
        else:
            print_text = 'Succeed! Find {}/{} 2D-3D inliers,q_err: {:.2f}, t_err: {:.2f}'.format(
                num_inliers, num_matches, q_err, t_err)
            print(print_text)
            q_frame.tracking_status = True

            self.update_current_frame(curr_frame=q_frame, ret=ret)
            return True

    def update_current_frame(self, curr_frame: Frame, ret: dict):
        curr_frame.qvec = ret['qvec']
        curr_frame.tvec = ret['tvec']

        curr_frame.matched_scene_name = ret['matched_scene_name']
        curr_frame.reference_frame_id = ret['reference_frame_id']
        inliers = np.array(ret['inliers'])

        curr_frame.matched_keypoints = ret['matched_keypoints'][inliers]
        curr_frame.matched_xyzs = ret['matched_xyzs'][inliers]
        curr_frame.matched_point3D_ids = ret['matched_point3D_ids'][inliers]
        curr_frame.matched_keypoint_ids = ret['matched_keypoint_ids'][inliers]
        curr_frame.matched_sids = ret['matched_sids'][inliers]

    def track_last_frame(self, curr_frame: Frame, last_frame: Frame):
        curr_kpts = curr_frame.keypoints[:, :2]
        curr_scores = curr_frame.keypoints[:, 2]
        curr_descs = curr_frame.descriptors
        curr_kpt_ids = np.arange(curr_kpts.shape[0])

        last_kpts = last_frame.keypoints[:, :2]
        last_scores = last_frame.keypoints[:, 2]
        last_descs = last_frame.descriptors
        last_xyzs = last_frame.xyzs
        last_point3D_ids = last_frame.point3D_ids
        last_sids = last_frame.seg_ids

        # '''
        indices = self.matcher({
            'descriptors0': torch.from_numpy(curr_descs)[None].cuda().float(),
            'keypoints0': torch.from_numpy(curr_kpts)[None].cuda().float(),
            'scores0': torch.from_numpy(curr_scores)[None].cuda().float(),
            'image_shape0': (1, 3, curr_frame.camera.width, curr_frame.camera.height),

            'descriptors1': torch.from_numpy(last_descs)[None].cuda().float(),
            'keypoints1': torch.from_numpy(last_kpts)[None].cuda().float(),
            'scores1': torch.from_numpy(last_scores)[None].cuda().float(),
            'image_shape1': (1, 3, last_frame.camera.width, last_frame.camera.height),
        })['matches0'][0].cpu().numpy()
        '''

        indices = self.nn_matcher({
            'descriptors0': torch.from_numpy(curr_descs.transpose()).float().cuda()[None],
            'descriptors1': torch.from_numpy(last_descs.transpose()).float().cuda()[None],
        })['matches0'][0].cpu().numpy()
        '''

        valid = (indices >= 0)

        matched_point3D_ids = last_point3D_ids[indices[valid]]
        point3D_mask = (matched_point3D_ids >= 0)
        matched_point3D_ids = matched_point3D_ids[point3D_mask]
        matched_sids = last_sids[indices[valid]][point3D_mask]

        matched_kpts = curr_kpts[valid][point3D_mask]
        matched_kpt_ids = curr_kpt_ids[valid][point3D_mask]
        matched_xyzs = last_xyzs[indices[valid]][point3D_mask]
        matched_last_kpts = last_kpts[indices[valid]][point3D_mask]

        print('Tracking: {:d} matches from {:d}-{:d} kpts'.format(matched_kpts.shape[0], curr_kpts.shape[0],
                                                                  last_kpts.shape[0]))

        # print('tracking: ', matched_kpts.shape, matched_xyzs.shape)
        ret = pycolmap.absolute_pose_estimation(matched_kpts + 0.5, matched_xyzs,
                                                curr_frame.camera,
                                                estimation_options={
                                                    "ransac": {"max_error": self.config['localization']['threshold']}},
                                                refinement_options={},
                                                # max_error_px=self.config['localization']['threshold']
                                                )
        if ret is None:
            ret = {'success': False, }
        else:
            ret['success'] = True
            ret['qvec'] = ret['cam_from_world'].rotation.quat[[3, 0, 1, 2]]
            ret['tvec'] = ret['cam_from_world'].translation

        ret['matched_keypoints'] = matched_kpts
        ret['matched_keypoint_ids'] = matched_kpt_ids
        ret['matched_ref_keypoints'] = matched_last_kpts
        ret['matched_xyzs'] = matched_xyzs
        ret['matched_point3D_ids'] = matched_point3D_ids
        ret['matched_sids'] = matched_sids
        ret['reference_frame_id'] = last_frame.reference_frame_id
        ret['matched_scene_name'] = last_frame.matched_scene_name
        return ret

    def track_last_frame_fast(self, curr_frame: Frame, last_frame: Frame):
        curr_kpts = curr_frame.keypoints[:, :2]
        curr_scores = curr_frame.keypoints[:, 2]
        curr_descs = curr_frame.descriptors
        curr_kpt_ids = np.arange(curr_kpts.shape[0])

        last_point3D_ids = last_frame.point3D_ids
        point3D_mask = (last_point3D_ids >= 0)
        last_kpts = last_frame.keypoints[:, :2][point3D_mask]
        last_scores = last_frame.keypoints[:, 2][point3D_mask]
        last_descs = last_frame.descriptors[point3D_mask]
        last_xyzs = last_frame.xyzs[point3D_mask]
        last_sids = last_frame.seg_ids[point3D_mask]

        minx = np.min(last_kpts[:, 0])
        maxx = np.max(last_kpts[:, 0])
        miny = np.min(last_kpts[:, 1])
        maxy = np.max(last_kpts[:, 1])
        curr_mask = (curr_kpts[:, 0] >= minx) * (curr_kpts[:, 0] <= maxx) * (curr_kpts[:, 1] >= miny) * (
                curr_kpts[:, 1] <= maxy)

        curr_kpts = curr_kpts[curr_mask]
        curr_scores = curr_scores[curr_mask]
        curr_descs = curr_descs[curr_mask]
        curr_kpt_ids = curr_kpt_ids[curr_mask]
        # '''
        indices = self.matcher({
            'descriptors0': torch.from_numpy(curr_descs)[None].cuda().float(),
            'keypoints0': torch.from_numpy(curr_kpts)[None].cuda().float(),
            'scores0': torch.from_numpy(curr_scores)[None].cuda().float(),
            'image_shape0': (1, 3, curr_frame.camera.width, curr_frame.camera.height),

            'descriptors1': torch.from_numpy(last_descs)[None].cuda().float(),
            'keypoints1': torch.from_numpy(last_kpts)[None].cuda().float(),
            'scores1': torch.from_numpy(last_scores)[None].cuda().float(),
            'image_shape1': (1, 3, last_frame.camera.width, last_frame.camera.height),
        })['matches0'][0].cpu().numpy()
        '''

        indices = self.nn_matcher({
            'descriptors0': torch.from_numpy(curr_descs.transpose()).float().cuda()[None],
            'descriptors1': torch.from_numpy(last_descs.transpose()).float().cuda()[None],
        })['matches0'][0].cpu().numpy()
        '''

        valid = (indices >= 0)

        matched_point3D_ids = last_point3D_ids[indices[valid]]
        matched_sids = last_sids[indices[valid]]

        matched_kpts = curr_kpts[valid]
        matched_kpt_ids = curr_kpt_ids[valid]
        matched_xyzs = last_xyzs[indices[valid]]
        matched_last_kpts = last_kpts[indices[valid]]

        print('Tracking: {:d} matches from {:d}-{:d} kpts'.format(matched_kpts.shape[0], curr_kpts.shape[0],
                                                                  last_kpts.shape[0]))

        # print('tracking: ', matched_kpts.shape, matched_xyzs.shape)
        ret = pycolmap.absolute_pose_estimation(matched_kpts + 0.5, matched_xyzs,
                                                curr_frame.camera._asdict(),
                                                max_error_px=self.config['localization']['threshold'])

        ret['matched_keypoints'] = matched_kpts
        ret['matched_keypoint_ids'] = matched_kpt_ids
        ret['matched_ref_keypoints'] = matched_last_kpts
        ret['matched_xyzs'] = matched_xyzs
        ret['matched_point3D_ids'] = matched_point3D_ids
        ret['matched_sids'] = matched_sids
        ret['reference_frame_id'] = last_frame.reference_frame_id
        ret['matched_scene_name'] = last_frame.matched_scene_name
        return ret

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

        ids1 = np.arange(matches.shape[0])
        ids2 = matches
        ids1 = ids1[matches >= 0]
        ids2 = ids2[matches >= 0]

        mask_p3ds = reference_frame.points3d_mask[ids2]
        ids1 = ids1[mask_p3ds]
        ids2 = ids2[mask_p3ds]

        return ids1, ids2
