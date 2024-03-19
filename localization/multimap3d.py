# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> multimap3d
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 13:47
=================================================='''
import numpy as np
import os
import os.path as osp
import time
import cv2
import torch
import yaml
from copy import deepcopy
from recognition.vis_seg import vis_seg_point, generate_color_dic, vis_inlier, plot_matches
from localization.base_model import dynamic_load
import localization.matchers as matchers
from localization.match_features import confs as matcher_confs
from nets.gm import GM
from tools.common import resize_img
from localization.singlemap3d import SingleMap3D
from localization.frame import Frame


class MultiMap3D:
    def __init__(self, config, viewer=None, save_dir=None):
        self.config = config
        self.save_dir = save_dir

        self.scenes = []
        self.sid_scene_name = []
        self.sub_maps = {}
        self.scene_name_start_sid = {}

        self.loc_config = config['localization']
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.matching_method = config['localization']['matching_method']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = dynamic_load(matchers, self.matching_method)
        self.matcher = Model(matcher_confs[self.matching_method]['model']).eval().to(device)

        self.initialize_map(config=config)
        self.loc_config = config['localization']

        self.viewer = viewer

        # options
        self.do_refinement = self.loc_config['do_refinement']
        self.refinement_method = self.loc_config['refinement_method']
        self.semantic_matching = self.loc_config['semantic_matching']
        self.do_pre_filtering = self.loc_config['pre_filtering_th'] > 0
        self.pre_filtering_th = self.loc_config['pre_filtering_th']

    def initialize_map(self, config):
        n_class = 0
        datasets = config['dataset']

        for name in datasets:
            config_path = osp.join(config['config_path'], '{:s}.yaml'.format(name))
            dataset_name = name

            with open(config_path, 'r') as f:
                scene_config = yaml.load(f, Loader=yaml.Loader)

            scenes = scene_config['scenes']
            for sid, scene in enumerate(scenes):
                self.scenes.append(name + '/' + scene)

                new_config = deepcopy(config)
                new_config['dataset_path'] = osp.join(config['dataset_path'], dataset_name, scene)
                new_config['segment_path'] = osp.join(config['segment_path'], dataset_name, scene)
                new_config['n_cluster'] = scene_config[scene]['n_cluster']
                new_config['cluster_mode'] = scene_config[scene]['cluster_mode']
                new_config['cluster_method'] = scene_config[scene]['cluster_method']
                new_config['gt_pose_path'] = scene_config[scene]['gt_pose_path']
                new_config['image_path_prefix'] = scene_config[scene]['image_path_prefix']
                sub_map = SingleMap3D(config=new_config,
                                      matcher=self.matcher,
                                      with_compress=config['localization']['with_compress'],
                                      start_sid=n_class)
                self.sub_maps[dataset_name + '/' + scene] = sub_map

                n_scene_class = scene_config[scene]['n_cluster']
                self.sid_scene_name = self.sid_scene_name + [dataset_name + '/' + scene for ni in range(n_scene_class)]
                self.scene_name_start_sid[dataset_name + '/' + scene] = n_class
                n_class = n_class + n_scene_class

                # break
        print('Load {} sub_maps from {} datasets'.format(len(self.sub_maps), len(datasets)))

    def run(self, q_frame: Frame):
        show = self.loc_config['show']
        seg_color = generate_color_dic(n_seg=2000)
        if show:
            cv2.namedWindow('loc', cv2.WINDOW_NORMAL)

        q_loc_segs = self.process_segmentations(segs=torch.from_numpy(q_frame.segmentations),
                                                topk=self.loc_config['seg_k'])
        q_pred_segs_top1 = q_frame.seg_ids  # initial results

        q_scene_name = q_frame.scene_name
        q_name = q_frame.name
        q_full_name = osp.join(q_scene_name, q_name)

        q_loc_sids = {}
        for v in q_loc_segs:
            q_loc_sids[v[0]] = (v[1], v[2])
        query_sids = list(q_loc_sids.keys())

        for i, sid in enumerate(query_sids):
            t_start = time.time()
            q_kpt_ids = q_loc_sids[sid][0]
            print(q_scene_name, q_name, sid)

            sid = sid - 1  # start from 0, confused!

            pred_scene_name = self.sid_scene_name[sid]
            start_seg_id = self.scene_name_start_sid[pred_scene_name]
            pred_sid_in_sub_scene = sid - self.scene_name_start_sid[pred_scene_name]
            pred_sub_map = self.sub_maps[pred_scene_name]
            pred_image_path_prefix = pred_sub_map.image_path_prefix

            print('pred/gt scene: {:s}, {:s}, sid: {:d}'.format(pred_scene_name, q_scene_name, pred_sid_in_sub_scene))
            print('{:s}/{:s}, pred: {:s}, sid: {:d}, order: {:d}'.format(q_scene_name, q_name, pred_scene_name, sid,
                                                                         i))

            if (q_kpt_ids.shape[0] >= self.loc_config['min_kpts']
                    and self.semantic_matching
                    and pred_sub_map.check_semantic_consistency(q_frame=q_frame,
                                                                sid=pred_sid_in_sub_scene,
                                                                overlap_ratio=0.5)):
                semantic_matching = True
            else:
                q_kpt_ids = np.arange(q_frame.keypoints.shape[0])
                semantic_matching = False
            print_text = f'Semantic matching - {semantic_matching}! Query kpts {q_kpt_ids.shape[0]} for {i}th seg {sid}'
            print(print_text)
            ret = pred_sub_map.localize_with_ref_frame(q_frame=q_frame,
                                                       q_kpt_ids=q_kpt_ids,
                                                       sid=pred_sid_in_sub_scene,
                                                       semantic_matching=semantic_matching)

            q_frame.time_loc = q_frame.time_loc + time.time() - t_start  # accumulate tracking time

            if show:
                reference_frame = pred_sub_map.reference_frames[ret['reference_frame_id']]
                ref_img = cv2.imread(osp.join(self.config['dataset_path'], pred_scene_name, pred_image_path_prefix,
                                              reference_frame.name))
                q_img_seg = vis_seg_point(img=q_frame.image, kpts=q_frame.keypoints[q_kpt_ids, :2],
                                          segs=q_frame.seg_ids[q_kpt_ids] + 1,
                                          seg_color=seg_color)
                matched_points3D_ids = ret['matched_point3D_ids']
                ref_sids = np.array([pred_sub_map.point3Ds[v].seg_id for v in matched_points3D_ids]) + \
                           self.scene_name_start_sid[pred_scene_name] + 1  # start from 1 as bg is 0
                ref_img_seg = vis_seg_point(img=ref_img, kpts=ret['matched_ref_keypoints'], segs=ref_sids,
                                            seg_color=seg_color)
                q_matched_kpts = ret['matched_keypoints']
                ref_matched_kpts = ret['matched_ref_keypoints']
                img_loc_matching = plot_matches(img1=q_img_seg, img2=ref_img_seg,
                                                pts1=q_matched_kpts, pts2=ref_matched_kpts,
                                                inliers=np.array([True for i in range(q_matched_kpts.shape[0])]),
                                                radius=9, line_thickness=3
                                                )

                q_frame.image_matching = img_loc_matching

                q_ref_img_matching = np.hstack([resize_img(q_img_seg, nh=512),
                                                resize_img(ref_img_seg, nh=512),
                                                resize_img(img_loc_matching, nh=512)])

            ret['order'] = i
            ret['matched_scene_name'] = pred_scene_name
            if not ret['success']:
                num_matches = ret['matched_keypoints'].shape[0]
                num_inliers = ret['num_inliers']
                print_text = f'Localization failed with {num_matches}/{q_kpt_ids.shape[0]} matches and {num_inliers} inliers, order {i}'
                print(print_text)

                if show:
                    show_text = 'FAIL! order: {:d}/{:d}-{:d}/{:d}'.format(i, len(q_loc_segs),
                                                                          num_matches,
                                                                          q_kpt_ids.shape[0])
                    q_img_inlier = vis_inlier(img=q_img_seg, kpts=ret['matched_keypoints'], inliers=ret['inliers'],
                                              radius=9 + 2, thickness=2)
                    q_img_inlier = cv2.putText(img=q_img_inlier, text=show_text, org=(30, 30),
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                               thickness=2, lineType=cv2.LINE_AA)
                    q_img_loc = np.hstack([resize_img(q_ref_img_matching, nh=512), resize_img(q_img_inlier, nh=512)])
                    cv2.imshow('loc', q_img_loc)
                    key = cv2.waitKey(self.loc_config['show_time'])
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        exit(0)
                continue

            success = self.verify_and_update(q_frame=q_frame, ret=ret)
            if show:
                q_err, t_err = q_frame.compute_pose_error()
                num_matches = ret['matched_keypoints'].shape[0]
                num_inliers = ret['num_inliers']
                show_text = 'order: {:d}/{:d}, k/m/i: {:d}/{:d}/{:d}'.format(
                    i, len(q_loc_segs), q_kpt_ids.shape[0], num_matches, num_inliers)
                q_img_inlier = vis_inlier(img=q_img_seg, kpts=ret['matched_keypoints'], inliers=ret['inliers'],
                                          radius=9 + 2, thickness=2)
                q_img_inlier = cv2.putText(img=q_img_inlier, text=show_text, org=(30, 30),
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                           thickness=2, lineType=cv2.LINE_AA)
                show_text = 'r_err:{:.2f}, t_err:{:.2f}'.format(q_err, t_err)
                q_img_inlier = cv2.putText(img=q_img_inlier, text=show_text, org=(30, 80),
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                           thickness=2, lineType=cv2.LINE_AA)
                q_frame.image_inlier = q_img_inlier

                q_img_loc = np.hstack([resize_img(q_ref_img_matching, nh=512), resize_img(q_img_inlier, nh=512)])

                cv2.imshow('loc', q_img_loc)
                key = cv2.waitKey(self.loc_config['show_time'])
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(0)

            if not success:
                continue
            else:
                break

        if q_frame.tracking_status is None:
            print('Failed to find a proper reference frame.')
            return False

        # do refinement
        if not self.do_refinement:
            return True
        else:
            t_start = time.time()
            pred_sub_map = self.sub_maps[q_frame.matched_scene_name]
            if q_frame.tracking_status is True and np.sum(q_frame.matched_inliers) >= 64:
                ret = pred_sub_map.refine_pose(q_frame=q_frame, refinement_method=self.loc_config['refinement_method'])
            else:
                ret = pred_sub_map.refine_pose(q_frame=q_frame,
                                               refinement_method='matching')  # do not trust the pose for projection

            q_frame.time_ref = time.time() - t_start

            inlier_mask = np.array(ret['inliers'])

            q_frame.qvec = ret['qvec']
            q_frame.tvec = ret['tvec']
            q_frame.matched_keypoints = ret['matched_keypoints'][inlier_mask]
            q_frame.matched_keypoint_ids = ret['matched_keypoint_ids'][inlier_mask]
            q_frame.matched_xyzs = ret['matched_xyzs'][inlier_mask]
            q_frame.matched_point3D_ids = ret['matched_point3D_ids'][inlier_mask]
            q_frame.matched_sids = ret['matched_sids'][inlier_mask]
            q_frame.matched_inliers = np.array(ret['inliers'])[inlier_mask]

            q_frame.refinement_reference_frame_ids = ret['refinement_reference_frame_ids']
            q_frame.reference_frame_id = ret['reference_frame_id']

            q_err, t_err = q_frame.compute_pose_error()
            ref_full_name = q_frame.matched_scene_name + '/' + pred_sub_map.reference_frames[
                q_frame.reference_frame_id].name
            print_text = 'Localization of {:s} success with inliers {:d}/{:d} with ref_name: {:s}, order: {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                q_full_name, ret['num_inliers'], len(ret['inliers']), ref_full_name, q_frame.matched_order, q_err,
                t_err)
            print(print_text)

            if show:
                q_err, t_err = q_frame.compute_pose_error()
                num_matches = ret['matched_keypoints'].shape[0]
                num_inliers = ret['num_inliers']
                show_text = 'Ref:{:d}/{:d},r_err:{:.2f}/t_err:{:.2f}'.format(num_matches, num_inliers, q_err,
                                                                             t_err)
                q_img_inlier = cv2.putText(img=q_img_inlier, text=show_text, org=(30, 130),
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                           thickness=2, lineType=cv2.LINE_AA)
                q_frame.image_inlier = q_img_inlier

            return True

    def verify_and_update(self, q_frame: Frame, ret: dict):
        num_matches = ret['matched_keypoints'].shape[0]
        num_inliers = ret['num_inliers']
        if q_frame.matched_keypoints is None or np.sum(q_frame.matched_inliers) < num_inliers:
            self.update_query_frame(q_frame=q_frame, ret=ret)

        q_err, t_err = q_frame.compute_pose_error(pred_qvec=ret['qvec'], pred_tvec=ret['tvec'])

        if num_inliers < self.loc_config['min_inliers']:
            print_text = 'Failed due to insufficient {:d} inliers, order {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                ret['num_inliers'], ret['order'], q_err, t_err)
            print(print_text)
            q_frame.tracking_status = False
            return False
        else:
            print_text = 'Succeed! Find {}/{} 2D-3D inliers, order {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                num_inliers, num_matches, ret['order'], q_err, t_err)
            print(print_text)
            q_frame.tracking_status = True
            return True

    def update_query_frame(self, q_frame, ret):
        q_frame.matched_scene_name = ret['matched_scene_name']
        q_frame.reference_frame_id = ret['reference_frame_id']
        q_frame.qvec = ret['qvec']
        q_frame.tvec = ret['tvec']

        inlier_mask = np.array(ret['inliers'])
        q_frame.matched_keypoints = ret['matched_keypoints']
        q_frame.matched_keypoint_ids = ret['matched_keypoint_ids']
        q_frame.matched_xyzs = ret['matched_xyzs']
        q_frame.matched_point3D_ids = ret['matched_point3D_ids']
        q_frame.matched_sids = ret['matched_sids']
        q_frame.matched_inliers = np.array(ret['inliers'])
        q_frame.matched_order = ret['order']

        # inlier_mask = np.array(ret['inliers'])
        # q_frame.matched_keypoints = ret['matched_keypoints'][inlier_mask]
        # q_frame.matched_keypoint_ids = ret['matched_keypoint_ids'][inlier_mask]
        # q_frame.matched_xyzs = ret['matched_xyzs'][inlier_mask]
        # q_frame.matched_point3D_ids = ret['matched_point3D_ids'][inlier_mask]
        # q_frame.matched_sids = ret['matched_sids'][inlier_mask]
        # q_frame.matched_inliers = np.array(ret['inliers'])[inlier_mask]

        # print('update_query_frame: ', q_frame.matched_keypoint_ids.shape, q_frame.matched_keypoints.shape,
        #       q_frame.matched_xyzs.shape, q_frame.matched_xyzs.shape, np.sum(q_frame.matched_inliers))

    def process_segmentations(self, segs, topk=10):
        pred_values, pred_ids = torch.topk(segs, k=segs.shape[-1], largest=True, dim=-1)  # [N, C]
        pred_values = pred_values.numpy()
        pred_ids = pred_ids.numpy()

        out = []
        used_sids = []
        for k in range(segs.shape[-1]):
            values_k = pred_values[:, k]
            ids_k = pred_ids[:, k]
            uids = np.unique(ids_k)

            out_k = []
            for sid in uids:
                if sid == 0:
                    continue
                if sid in used_sids:
                    continue
                used_sids.append(sid)
                ids = np.where(ids_k == sid)[0]
                score = np.mean(values_k[ids])
                # score = np.median(values_k[ids])
                # score = 100 - k
                # out_k.append((ids.shape[0], sid - 1, ids, score))
                out_k.append((ids.shape[0], sid, ids, score))

            out_k = sorted(out_k, key=lambda item: item[0], reverse=True)
            for v in out_k:
                out.append((v[1], v[2], v[3]))  # [sid, ids, score]
                if len(out) >= topk:
                    return out
        return out
