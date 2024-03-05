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
import pycolmap
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
from localization.refframe import RefFrame
from localization.utils import compute_pose_error


class MultiMap3D:
    def __init__(self, config, viewer=None, save_dir=None):
        self.config = config
        self.save_dir = save_dir

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
        self.scenes = []
        self.sid_scene_name = []
        self.sub_maps = {}
        self.scene_name_start_sid = {}
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
                                      with_compress=config['localization']['with_compress'])
                self.sub_maps[dataset_name + '/' + scene] = sub_map

                n_scene_class = scene_config[scene]['n_cluster']
                self.sid_scene_name = self.sid_scene_name + [dataset_name + '/' + scene for ni in range(n_scene_class)]
                self.scene_name_start_sid[dataset_name + '/' + scene] = n_class
                n_class = n_class + n_scene_class

                break
        print('Load {} sub_maps from {} datasets'.format(len(self.sub_maps), len(datasets)))

    def run(self, q_frame: Frame, q_segs: torch.Tensor):
        q_seg_scores = torch.softmax(q_segs, dim=-1)  # [N, C]
        if self.do_pre_filtering:
            non_bg_mask = q_frame.filter_keypoints(seg_scores=q_seg_scores.cpu().numpy(),
                                                   filtering_threshold=self.pre_filtering_th)
            if non_bg_mask is not None:
                q_seg_scores = q_seg_scores[torch.from_numpy(non_bg_mask).cuda()]
                q_segs = q_segs[torch.from_numpy(non_bg_mask).cuda()]

        q_loc_segs = self.process_segmentations(segs=q_seg_scores, topk=self.loc_config['seg_k'])
        q_pred_segs_top1 = q_segs.max(dim=-1)[1]
        q_pred_segs_top1 = q_pred_segs_top1.cpu().numpy()

    def run1(self, q_frame: Frame, q_segs: torch.Tensor):
        t_loc = 0
        t_ref = 0
        img_loc_matching = None
        seg_color = generate_color_dic(n_seg=2000)
        self.loc_text = []

        show = self.loc_config['show']
        if show:
            cv2.namedWindow('loc', cv2.WINDOW_NORMAL)

        q_name = q_frame.name
        q_img = q_frame.image
        q_descs = q_frame.descriptors
        q_kpts = q_frame.keypoints[:, :2]
        q_scores = q_frame.keypoints[:, 2]
        q_scene_name = q_frame.scene_name

        q_seg_scores = torch.softmax(q_segs, dim=-1)  # [N, C]
        if self.do_pre_filtering:
            scores_background = q_seg_scores[:, 0]
            non_bg_mask = (scores_background < self.pre_filtering_th)
            print('pre filtering before: ', q_segs.shape)
            if torch.sum(non_bg_mask) >= 0.4 * q_seg_scores.shape[0]:
                q_seg_scores = q_seg_scores[non_bg_mask]
                non_bg_mask = non_bg_mask.cpu().numpy()
                q_descs = q_descs[non_bg_mask]
                q_kpts = q_kpts[non_bg_mask]
                q_scores = q_scores[non_bg_mask]
                q_segs = q_segs[non_bg_mask]
            print('pre filtering after: ', q_segs.shape)

        q_loc_segs = self.process_segmentations(segs=q_seg_scores, topk=self.loc_config['seg_k'])
        q_pred_segs_top1 = q_segs.max(dim=-1)[1]
        q_pred_segs_top1 = q_pred_segs_top1.cpu().numpy()

        log_text = ['qname: {:s} with {:d} kpts'.format(q_name, q_kpts.shape[0])]
        best_result = None
        gt_sub_map = self.sub_maps[q_scene_name]
        if gt_sub_map.gt_poses is not None and q_name in gt_sub_map.gt_poses.keys():
            gt_qcw = gt_sub_map.gt_poses[q_name]['qvec']
            gt_tcw = gt_sub_map.gt_poses[q_name]['tvec']
        else:
            gt_qcw = None
            gt_tcw = None
        gt_sub_map = self.sub_maps[q_scene_name]
        q_loc_sids = {}
        for v in q_loc_segs:
            q_loc_sids[v[0]] = (v[1], v[2])

        query_sids = list(q_loc_sids.keys())
        q_full_name = osp.join(q_scene_name, q_name)

        t_start = time.time()

        for i, sid in enumerate(query_sids):
            q_seg_ids = q_loc_sids[sid][0]

            print(q_scene_name, q_name, sid)

            pred_scene_name = self.sid_scene_name[sid]
            start_seg_id = self.scene_name_start_sid[pred_scene_name]
            pred_sid_in_sub_scene = sid - self.scene_name_start_sid[pred_scene_name]
            pred_sub_map = self.sub_maps[pred_scene_name]
            pred_image_path_prefix = pred_sub_map.image_path_prefix

            print('pred/gt scene: {:s}, {:s}, sid: {:d}'.format(pred_scene_name, q_scene_name, pred_sid_in_sub_scene))

            print('{:s}/{:s}, pred: {:s}, sid: {:d}, order: {:d}'.format(q_scene_name, q_name, pred_scene_name, sid,
                                                                         i))
            if q_seg_ids.shape[0] >= self.loc_config['min_kpts'] and self.semantic_matching:
                print_text = 'Semantic matching! Query kpts {} for {}th seg {}'.format(q_seg_ids.shape[0], i, sid)
                print(print_text)
                log_text.append(print_text)

                q_seg_descs = q_descs[q_seg_ids]
                q_seg_kpts = q_kpts[q_seg_ids]
                q_seg_scores = q_scores[q_seg_ids]
                q_seg_sid_top1 = q_pred_segs_top1[q_seg_ids]
                semantic_matching = True
            else:
                print_text = 'Not semantic matching! Query kpts {} for {}th seg {}, use all kpts'.format(
                    q_kpts.shape[0], i, sid)
                print(print_text)
                log_text.append(print_text)

                q_seg_descs = q_descs
                q_seg_kpts = q_kpts
                q_seg_scores = q_scores
                q_seg_sid_top1 = q_pred_segs_top1
                semantic_matching = False

            query_data = {
                'descriptors': q_seg_descs,
                'scores': q_seg_scores,
                'keypoints': q_seg_kpts,
                'camera': q_frame.camera,
            }

            ret = pred_sub_map.localize_with_ref_frame(query_data=query_data,
                                                       sid=pred_sid_in_sub_scene,
                                                       semantic_matching=semantic_matching)
            n_matches = ret['matched_keypoints'].shape[0]
            n_inliers = ret['num_inliers']
            inliers = ret['inliers']
            success = ret['success']
            ref_frame_id = ret['ref_frame_id']
            if not success:
                print_text = 'Localization failed with {}/{} matches and {} inliers, order {}'.format(
                    n_matches,
                    q_seg_descs.shape[0],
                    n_inliers, i)
                print(print_text)
                log_text.append(print_text)

            if gt_qcw is not None:
                q_err, t_err = compute_pose_error(pred_qcw=ret['qvec'],
                                                  pred_tcw=ret['tvec'],
                                                  gt_qcw=gt_qcw,
                                                  gt_tcw=gt_tcw)
            else:
                q_err = 1e2
                t_err = 1e2

            if n_inliers < self.loc_config['min_inliers']:
                print_text = 'qname: {:s} failed due to insufficient {:d} inliers, order {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                    q_name,
                    ret['num_inliers'],
                    i, q_err, t_err)
                print(print_text)
                log_text.append(print_text)

                update_best_result = False
                if best_result is None:
                    update_best_result = True
                elif best_result['num_inliers'] < n_inliers:
                    update_best_result = True

                if update_best_result:
                    best_result['num_inliers'] = ret['num_inliers']
                    best_result['qvec'] = ret['qvec']
                    best_result['tvec'] = ret['tvec']
                    best_result['order'] = i
                    best_result['scene_name'] = pred_scene_name
                    best_result['seg_id'] = sid

                continue
            else:
                loc_success = True
                print_text = 'Find {}/{} 2D-3D inliers'.format(n_inliers, len(inliers))
                print(print_text)
                log_text.append(print_text)

                if gt_qcw is not None:
                    q_err, t_err = compute_pose_error(pred_qcw=ret['qvec'],
                                                      pred_tcw=ret['tvec'],
                                                      gt_qcw=gt_qcw,
                                                      gt_tcw=gt_tcw)
                    print_text = 'qname: {:s} r_err: {:.2f}, t_err: {:.2f}'.format(q_name, q_err, t_err)
                    print(print_text)
                    log_text.append(print_text)
                else:
                    q_err = 1e2
                    t_err = 1e2

                if self.do_refinement:
                    t_start = time.time()
                    query_data = {
                        'camera': q_frame.camera,
                        'descriptors': q_descs,
                        'scores': q_scores,
                        'keypoints': q_kpts,
                        'qvec': ret['qvec'],
                        'tvec': ret['tvec'],
                        'n_inliers': n_inliers,
                        'loc_success': loc_success,
                        'matched_keypoints': ret['matched_keypoints'],
                        'matched_xyzs': ret['matched_xyzs'],
                    }
                    ref_ret = pred_sub_map.refine_pose(query_data=query_data, ref_frame_id=ret['ref_frame_id'],
                                                       refinement_method=self.refinement_method)
                    t_ref = time.time() - t_start
                    if ref_ret['success']:
                        if gt_qcw is not None:
                            q_err, t_err = compute_pose_error(pred_qcw=ref_ret['qvec'],
                                                              pred_tcw=ref_ret['tvec'],
                                                              gt_qcw=gt_qcw,
                                                              gt_tcw=gt_tcw)
                        else:
                            q_err, t_err = 1e2, 1e2

                        best_ref_full_name = pred_scene_name + '/' + pred_sub_map.ref_frames[ref_frame_id].name
                        print_text = 'Localization of {:s} success with inliers {:d}/{:d} with ref_name: {:s}, order: {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                            q_full_name, ref_ret['num_inliers'], len(ref_ret['inliers']), best_ref_full_name, i, q_err,
                            t_err)
                        print(print_text)
                        log_text.append(print_text)
                        out = {
                            'qvec': ref_ret['qvec'],
                            'tvec': ref_ret['tvec'],
                            'matched_keypoints': ret['matched_keypoints'],
                            'matched_xyzs': ret['matched_xyzs'],
                            'success': True,
                            'log': log_text,
                            'q_err': q_err,
                            't_err': t_err,
                            'time_loc': t_loc,
                            'time_ref': t_ref,
                        }

                        return out
                    else:
                        continue
                else:
                    out = {
                        'gt_qvec': gt_qcw,
                        'gt_tvec': gt_tcw,
                        'qvec': ret['qvec'],
                        'tvec': ret['tvec'],
                        'inliers': np.array(ret['inliers']),
                        'log': log_text,
                        'n_inliers': len(inliers),
                        'success': True,
                        'q_err': q_err,
                        't_err': t_err,
                        'time_ref': t_ref,
                        'time_loc': t_loc,
                        # 'img_loc': img_vis,
                        'query_img': q_img,
                        'img_matching': img_loc_matching,

                        'pred_sid': sid,
                        # 'reference_db_ids': [ref_img_id],
                        # 'vrf_image_id': ref_img_id,
                        'start_seg_id': start_seg_id,
                    }

                    return out

        # failed to find a good reference frame
        if best_result['num_inliers'] >= 4:
            best_pred_scene = best_result['scene_name']
            best_sub_map = self.sub_maps[best_pred_scene]
            print('Try to do localization from best results inliers {},ref_name {}, order {}'.format(
                best_result['num_inliers'],
                best_result['ref_img_name'],
                best_result['order']))
            if self.do_refinement:
                t_start = time.time()
                ref_ret = best_sub_map.refine_pose(
                    query_data={
                        'camera': q_frame.camera,
                        'descriptors': q_descs,
                        'scores': q_scores,
                        'keypoints': q_kpts,
                        'qvec': None,
                        'tvec': None,
                        'n_inliers': best_result['num_inliers'],
                        'loc_success': False,
                    },
                    ref_frame_id=best_result['ref_img_id'],
                )

                t_ref = time.time() - t_start

                if ref_ret['success']:
                    q_err, t_err = compute_pose_error(pred_qcw=ref_ret['qvec'],
                                                      pred_tcw=ref_ret['tvec'],
                                                      gt_qcw=gt_sub_map.gt_poses[q_name]['qvec'],
                                                      gt_tcw=gt_sub_map.gt_poses[q_name]['tvec'])
                else:
                    q_err, t_err = 1e2, 1e2

                best_ref_full_name = osp.join(best_pred_scene, best_result['ref_img_name'])
                print_text = 'Localization of {:s} success with inliers {:d}/{:d} ref_name: {:s}, order: {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                    q_full_name,
                    ref_ret['num_inliers'], len(ref_ret['inliers']),
                    best_ref_full_name, best_result['order'], q_err, t_err)
            print(print_text)
            log_text.append(print_text)
            out = {
                'qvec': ref_ret['qvec'],
                'tvec': ref_ret['tvec'],
                'success': False,
                'log': log_text,
                'q_err': q_err,
                't_err': t_err,
                'time_loc': t_loc,
                'time_ref': t_ref,
            }
            return out

        print_text = 'qname: {:s} find the best among all candidates'.format(q_name)
        log_text.append(print_text)

        if gt_sub_map.gt_poses is not None and q_name in gt_sub_map.gt_poses.keys():
            q_err, t_err = compute_pose_error(pred_qcw=best_result['qvec'],
                                              pred_tcw=best_result['tvec'],
                                              gt_qcw=gt_sub_map.gt_poses[q_name]['qvec'],
                                              gt_tcw=gt_sub_map.gt_poses[q_name]['tvec'])
        else:
            q_err, t_err = 1e2, 1e2

        out = {
            'gt_qvec': gt_qcw,
            'gt_tvec': gt_tcw,
            'qvec': best_result['qvec'],
            'tvec': best_result['tvec'],
            'success': False,
            'log': log_text,
            'q_err': q_err,
            't_err': t_err,

            'time_loc': t_loc,
            'time_ref': t_ref,
        }
        return out

    def localize_from_sid(self, order, sid: int, q_kpts, q_scores, q_descs, q_pred_segs_top1, q_seg_ids):
        pred_scene_name = self.sid_scene_name[sid]
        start_seg_id = self.scene_name_start_sid[pred_scene_name]
        pred_sid_in_sub_scene = sid - self.scene_name_start_sid[pred_scene_name]
        pred_sub_map = self.sub_maps[pred_scene_name]

        # no corresponding vrf for sid
        if len(pred_sub_map.seg_vrf[pred_sid_in_sub_scene].keys() == 0):
            return None

        if q_seg_ids.shape[0] >= self.loc_config['min_kpts'] and self.semantic_matching:
            print_text = 'Semantic matching! Query kpts {} for {}th seg {}'.format(q_seg_ids.shape[0], order, sid)
            print(print_text)
            self.log_text.append(print_text)

            q_seg_descs = q_descs[q_seg_ids]
            q_seg_kpts = q_kpts[q_seg_ids]
            q_seg_scores = q_scores[q_seg_ids]
            q_seg_sid_top1 = q_pred_segs_top1[q_seg_ids]
            seg_wise_matching = True

        else:
            print_text = 'Not semantic matching! Query kpts {} for {}th seg {}, use all kpts'.format(
                q_kpts.shape[0], order, sid)
            print(print_text)
            self.log_text.append(print_text)

            q_seg_descs = q_descs
            q_seg_kpts = q_kpts
            q_seg_scores = q_scores
            q_seg_sid_top1 = q_pred_segs_top1
            seg_wise_matching = False

        ref_frame = pred_sub_map.ref_frames[pred_sub_map.seg_vrf[pred_sid_in_sub_scene][0]]

    def process_segmentations(self, segs, topk=10):
        pred_values, pred_ids = torch.topk(segs, k=segs.shape[-1], largest=True, dim=-1)  # [N, C]
        pred_values = pred_values.cpu().numpy()
        pred_ids = pred_ids.cpu().numpy()

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
                out_k.append((ids.shape[0], sid - 1, ids, score))

            out_k = sorted(out_k, key=lambda item: item[0], reverse=True)
            for v in out_k:
                out.append((v[1], v[2], v[3]))  # [sid, ids, score]
                if len(out) >= topk:
                    return out
        return out
