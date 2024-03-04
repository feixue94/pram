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
from localization.simglelocmap import SingleLocMap
from nets.gm import GM
from tools.common import resize_img
from localization.singlemap3d import SingleMap3D
from localization.frame import Frame
from localization.refframe import RefFrame


class MultiMap3D:
    def __init__(self, config, viewer=None, save_dir=None):
        self.config = config
        self.save_dir = save_dir

        self.loc_config = config['localization']
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

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
                sub_map = SingleMap3D(config=new_config, with_compress=config['localization']['with_compress'])
                self.sub_maps[dataset_name + '/' + scene] = sub_map

                n_scene_class = scene_config[scene]['n_cluster']
                self.sid_scene_name = self.sid_scene_name + [dataset_name + '/' + scene for ni in range(n_scene_class)]
                self.scene_name_start_sid[dataset_name + '/' + scene] = n_class
                n_class = n_class + n_scene_class
        print('Load {} sub_maps from {} datasets'.format(len(self.sub_maps), len(datasets)))

    def run(self, q_frame: Frame, q_segs: torch.Tensor):
        t_loc = 0
        t_ref = 0
        img_loc = None
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
        q_pred_segs_top1 = torch.from_numpy(q_segs).cuda().max(dim=-1)[1]
        q_pred_segs_top1 = q_pred_segs_top1.cpu().numpy()

        log_text = ['qname: {:s} with {:d} kpts'.format(q_name, q_kpts.shape[0])]
        best_results = {
            'num_inliers': -1,
            'ref_img_name': '',
            'ref_img_id': -1,
            'qvec': None,
            'tvec': None,
            'order': -1,
            'scene_name': None,
            'seg_id': -1,
        }

        gt_sub_map = self.sub_maps[q_scene_name]
        q_loc_sids = {}
        for v in q_loc_segs:
            q_loc_sids[v[0]] = (v[1], v[2])
        query_sids = list(q_loc_sids.keys())
        q_full_name = osp.join(q_scene_name, q_name)

        t_start = time.time()

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

    def match_between_query_and_reference(self, query_data: dict,
                                          pred_sub_map: SingleMap3D,
                                          ref_frame: RefFrame, pred_sid_in_scene: int,
                                          semantic_matching=False):
        pass

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
