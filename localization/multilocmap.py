# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> multilocmap
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 16:24
=================================================='''
import os
import os.path as osp
import torch
import yaml
from copy import deepcopy
import time, cv2
import numpy as np
import pycolmap
from localization.utils import compute_pose_error
from recognition.vis_seg import vis_seg_point, generate_color_dic, vis_inlier, plot_matches
from localization.base_model import dynamic_load
import localization.matchers as matchers
from localization.match_features import confs as matcher_confs
from localization.simglelocmap import SingleLocMap
from nets.gm import GM
from tools.common import resize_img


def process_segmentations(segs, topk=10):
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


class MultiLocMap:
    '''
    localization in multiple scenes (e.g., seven rooms in 7scenes together)
    '''

    def __init__(self, config, viewer=None, save_dir=None, desc_compressor=None):
        self.config = config
        self.save_dir = save_dir
        self.matching_method = config['localization']['matching_method']
        if self.matching_method in ['gm', 'gml']:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            Model = dynamic_load(matchers, self.matching_method)
            self.matcher = Model(matcher_confs[self.matching_method]['model']).eval().to(device)
        elif self.matching_method == 'gm':
            desc_dim = config['feat_dim']
            self.gm = GM(config={
                'descriptor_dim': desc_dim,
                'hidden_dim': 256,
                'keypoint_encoder': [32, 64, 128, 256],
                'GNN_layers': ['self', 'cross'] * 9,  # [self, cross, self, cross, ...] 9 in total
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
                'n_layers': 9,
                'with_sinkhorn': True,
                'ac_fn': 'relu',
                'norm_fn': 'bn',
            })
            if desc_dim == 128:
                weight_path = '/scratches/flyer_3/fx221/exp/uniloc/20230519_145436_gm_L9_resnet4x_B16_K1024_M0.2_relu_bn_adam/gm.900.pth'
            elif desc_dim == 64:
                weight_path = '/scratches/flyer_3/fx221/exp/uniloc/20230704_150352_gm64_L9_resnet4x_B16_K1024_M0.2_relu_bn_adam/gm64.900.pth'
            elif desc_dim == 32:
                weight_path = '/scratches/flyer_3/fx221/exp/uniloc/20230704_131643_gm32_L9_resnet4x_B16_K1024_M0.2_relu_bn_adam/gm32.850.pth'
            elif desc_dim == 0:
                weight_path = '/scratches/flyer_3/fx221/exp/uniloc/20230630_113805_gmc_L9_resnet4x_B16_K1024_M0.2_relu_bn_adam/gmc.370.pth'
            state_dict = torch.load(weight_path, map_location='cpu')['model']
            self.gm.load_state_dict(state_dict=state_dict, strict=True)
            self.gm.desc_compressor = desc_compressor
            self.gm.cuda().eval()
            self.matcher = self.gm
        else:
            self.gm = None

        self.initialize_map(config=config)

        self.loc_config = config['localization']
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.viewer = viewer

        # options
        self.do_refinement = self.loc_config['do_refinement']
        self.refinement_method = self.loc_config['refinement_method']
        self.semantic_matching = self.loc_config['semantic_matching']

    def set_viewer(self, viewer):
        self.viewer = viewer

    def initialize_map(self, config):
        self.scenes = []
        self.sid_scene_name = []
        self.sub_maps = {}
        self.scene_name_start_sid = {}
        n_class = 0
        datasets = config['dataset']
        for name in datasets:
            if name in ['Aachen', 'A']:
                config_path = osp.join(config['config_path'], 'Aachen.yaml')
                dataset_name = 'Aachen'
            elif name in ['RobotCar-Seasons', 'R']:
                config_path = osp.join(config['config_path'], 'RobotCar-Seasons.yaml')
                dataset_name = 'RobotCar-Seasons'
            elif name in ['CambridgeLandmarks', 'C']:
                config_path = osp.join(config['config_path'], 'CambridgeLandmarks.yaml')
                dataset_name = 'CambridgeLandmarks'
            elif name in ['7Scenes', 'S']:
                config_path = osp.join(config['config_path'], '7Scenes.yaml')
                dataset_name = '7Scenes'
            elif name in ['12Scenes', 'T']:
                config_path = osp.join(config['config_path'], '12Scenes.yaml')
                dataset_name = '12Scenes'
            else:
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
                sub_map = SingleLocMap(config=new_config, save_dir=self.save_dir, matcher=self.matcher,
                                       with_compress=config['localization']['with_compress'])
                self.sub_maps[dataset_name + '/' + scene] = sub_map

                n_scene_class = scene_config[scene]['n_cluster']
                self.sid_scene_name = self.sid_scene_name + [dataset_name + '/' + scene for ni in range(n_scene_class)]
                self.scene_name_start_sid[dataset_name + '/' + scene] = n_class
                n_class = n_class + n_scene_class
        print('Load {} sub_maps from {} datasets'.format(len(self.sub_maps), len(datasets)))

    def run(self, query_data):
        t_loc = 0
        t_ref = 0
        img_loc = None
        img_loc_matching = None
        vis_out = {}

        save = self.loc_config['save']
        show = self.loc_config['show']
        seg_color = generate_color_dic(n_seg=2000)
        if show:
            cv2.namedWindow('loc', cv2.WINDOW_NORMAL)

        q_img = query_data['image']
        img_vis = resize_img(q_img, nh=512)

        self.q_img = q_img
        q_name = query_data['name']
        q_info = query_data['info']
        q_descs = query_data['descriptors']
        q_scores = query_data['scores']
        q_kpts = query_data['keypoints']
        q_seg = query_data['segmentations']

        q_scene_name = query_data['scene_name']

        do_pre_filtering = True
        th_score_background = 0.95
        q_seg_scores = torch.softmax(torch.from_numpy(q_seg).cuda(), dim=-1)
        if do_pre_filtering:  # and q_seg.shape[0] > min_kpts:
            scores_background = q_seg_scores[:, 0]
            non_bg_mask = (scores_background < th_score_background)
            print('pre filtering before: ', q_seg.shape)
            if torch.sum(non_bg_mask) >= 0.4 * q_seg_scores.shape[0]:
                q_seg_scores = q_seg_scores[non_bg_mask]
                non_bg_mask = non_bg_mask.cpu().numpy()
                q_descs = q_descs[non_bg_mask]
                q_kpts = q_kpts[non_bg_mask]
                q_scores = q_scores[non_bg_mask]
                q_seg = q_seg[non_bg_mask]
            print('pre filtering after: ', q_seg.shape)

        q_loc_segs = process_segmentations(segs=q_seg_scores, topk=self.loc_config['seg_k'])
        q_pred_segs_top1 = torch.from_numpy(q_seg).cuda().max(dim=-1)[1]
        q_pred_segs_top1 = q_pred_segs_top1.cpu().numpy()

        # localize by recognition
        camera_model, width, height, params = q_info
        cfg = {
            'model': camera_model,
            'width': width,
            'height': height,
            'params': params,
        }

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
        if gt_sub_map.gt_poses is not None and q_name in gt_sub_map.gt_poses.keys():
            gt_qcw = gt_sub_map.gt_poses[q_name]['qvec']
            gt_tcw = gt_sub_map.gt_poses[q_name]['tvec']
        else:
            gt_qcw = None
            gt_tcw = None

        q_loc_sids = {}
        for v in q_loc_segs:
            q_loc_sids[v[0]] = (v[1], v[2])
        query_sids = list(q_loc_sids.keys())
        # query_sids = sorted_query_sids.keys()
        q_full_name = osp.join(q_scene_name, q_name)

        t_start = time.time()
        for i, sid in enumerate(query_sids):
            loc_success = False

            q_seg_ids = q_loc_sids[sid][0]

            print(q_scene_name, q_name, sid)

            pred_scene_name = self.sid_scene_name[sid]
            start_seg_id = self.scene_name_start_sid[pred_scene_name]
            pred_sid_in_sub_scene = sid - self.scene_name_start_sid[pred_scene_name]
            pred_sub_map = self.sub_maps[pred_scene_name]
            pred_image_path_prefix = pred_sub_map.image_path_prefix

            if len(pred_sub_map.seg_vrf[pred_sid_in_sub_scene].keys()) == 0:
                continue

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
                seg_wise_matching = True
            else:
                print_text = 'Not semantic matching! Query kpts {} for {}th seg {}, use all kpts'.format(
                    q_kpts.shape[0], i, sid)
                print(print_text)
                log_text.append(print_text)

                q_seg_descs = q_descs
                q_seg_kpts = q_kpts
                q_seg_scores = q_scores
                q_seg_sid_top1 = q_pred_segs_top1
                seg_wise_matching = False

            query_matching_data = {
                'descriptors': q_seg_descs,
                'scores': q_seg_scores,
                'keypoints': q_seg_kpts,
                'width': width,
                'height': height,
            }

            all_mp2ds = []
            all_mp3ds = []
            all_p3ds = []
            macth_vrf_info = {}
            max_vrf = 1
            mkq_p3d_ids = {}
            if self.loc_config['max_vrf'] > 1:
                max_vrf = min(self.loc_config['max_vrf'], len(pred_sub_map.seg_vrf[pred_sid_in_sub_scene].keys()))
            for vi in range(max_vrf):
                pred_vrf = pred_sub_map.seg_vrf[pred_sid_in_sub_scene][vi]
                match_out = self.build_matches_between_query_and_vrf(query_matching_data=query_matching_data,
                                                                     pred_sub_map=pred_sub_map,
                                                                     pred_vrf=pred_vrf,
                                                                     sid=sid,
                                                                     pred_sid_in_sub_scene=pred_sid_in_sub_scene,
                                                                     pred_scene_name=pred_scene_name,
                                                                     pred_image_path_prefix=pred_image_path_prefix,
                                                                     seg_wise_matching=seg_wise_matching)
                matches = match_out['matches']
                p3ds = match_out['p3ds']
                p3d_segs = match_out['p3d_segs']
                all_p3ds.append(p3ds)
                if np.sum(matches >= 0) > 0:
                    macth_vrf_info[vi] = {
                        'image_id': pred_vrf['image_id'],
                        'image_name': pred_vrf['image_name'],
                        'scene_name': pred_scene_name,
                        'start': len(all_mp2ds),
                        'end': len(all_mp2ds) + np.sum(matches >= 0),
                    }

                    matched_p3d_ids = match_out['p3d_ids']
                    for mi in range(matches.shape[0]):
                        if matches[mi] < 0:
                            continue
                        if mi in mkq_p3d_ids.keys():
                            if matched_p3d_ids[matches[mi]] in mkq_p3d_ids[mi]:
                                continue
                            else:
                                mkq_p3d_ids[mi].append(matched_p3d_ids[matches[mi]])
                        else:
                            mkq_p3d_ids[mi] = [matched_p3d_ids[matches[mi]]]

                    all_mp2ds.append(q_seg_kpts[matches >= 0])
                    all_mp3ds.append(p3ds[matches[matches >= 0]])

                print_text = 'Find {} 2D-3D ({}-{}) matches, vrf-id: {}, name: {}'.format(np.sum(matches >= 0),
                                                                                          q_seg_descs.shape[0],
                                                                                          p3ds.shape[0], vi,
                                                                                          pred_vrf['image_name'])
                print(print_text)
                log_text.append(print_text)

                # for visualization
                ref_img_id = pred_vrf['image_id']
                if show and 'ref_keypoints' in match_out.keys():
                    # build matches between query and map points
                    ref_img_name = pred_vrf['image_name']
                    ref_img_id = pred_vrf['image_id']
                    ref_img = cv2.imread(
                        osp.join(self.config['dataset_path'], pred_scene_name, pred_image_path_prefix, ref_img_name))

                    ref_kpts = match_out['ref_keypoints']
                    ref_mask = match_out['ref_mask']

                    valid_matches = (matches >= 0)

                    q_mkpts = q_seg_kpts[valid_matches]
                    ref_mkpts = ref_kpts[matches[valid_matches]]
                    ref_msegs = p3d_segs[matches[valid_matches]]

                    q_img_seg = vis_seg_point(img=q_img, kpts=q_seg_kpts,
                                              # segs=[sid + 1 for i in range(q_seg_kpts.shape[0])],
                                              # segs=[sid + 1 for i in range(q_seg_kpts.shape[0])],
                                              segs=q_seg_sid_top1,
                                              seg_color=seg_color)

                    ref_img = vis_seg_point(img=ref_img, kpts=ref_kpts[ref_mask],
                                            # segs=[sid + 1 for i in range(np.sum(ref_mask))],  # only for visualization
                                            segs=p3d_segs[ref_mask] + 1 + self.scene_name_start_sid[pred_scene_name],
                                            # only for visualization, this is a bug
                                            seg_color=seg_color)

                    img_loc_matching = plot_matches(img1=q_img_seg, img2=ref_img,
                                                    pts1=q_mkpts, pts2=ref_mkpts,
                                                    inliers=np.array([True for i in range(q_mkpts.shape[0])]),
                                                    radius=9,
                                                    line_thickness=3,
                                                    )
                    q_img_seg = resize_img(q_img_seg, nh=512)
                    ref_img = resize_img(img=ref_img, nh=512)
                    img_loc_matching = resize_img(img=img_loc_matching, nh=512)
                    q_ref_img = np.hstack([q_img_seg, ref_img, img_loc_matching])

            if len(all_mp2ds) > 0:
                all_mp2ds = np.vstack(all_mp2ds) + 0.5
                all_mp3ds = np.vstack(all_mp3ds)
                all_p3ds = np.vstack(all_p3ds)
                n_matches = all_mp2ds.shape[0]
                n_p3ds = all_p3ds.shape[0]
            else:
                n_matches = 0
                n_p3ds = 0

            if n_matches < self.loc_config['min_matches']:
                print_text = 'Insufficient matches {} between 2D-3D({}-{}) points, order {}'.format(n_matches,
                                                                                                    q_descs.shape[0],
                                                                                                    n_p3ds,
                                                                                                    i)
                print(print_text)
                log_text.append(print_text)
                continue
            else:
                print_text = 'Find {} 2D-3D ({}-{}) matches, order {}'.format(n_matches,
                                                                              q_seg_descs.shape[0],
                                                                              n_p3ds, i)
                print(print_text)
                log_text.append(print_text)

            # ret = pycolmap.absolute_pose_estimation(all_mp2ds, all_mp3ds, cfg, self.loc_config['threshold'])
            ret = pycolmap.absolute_pose_estimation(all_mp2ds, all_mp3ds, cfg, 12)

            t_loc = time.time() - t_start

            if not ret['success']:
                print_text = 'qname: {} failed after pose estimation'.format(q_name)
                print(print_text)
                log_text.append(print_text)

                if show:
                    img_loc = vis_seg_point(img=q_img, kpts=all_mp2ds,
                                            segs=np.zeros(shape=(all_mp2ds.shape[0],)) + sid, seg_color=seg_color,
                                            radius=9)
                    img_loc = resize_img(img=img_loc, nh=512)

                    show_text = 'FAIL! order: {:d}/{:d}'.format(i, len(q_loc_segs), q_seg_kpts.shape[0],
                                                                np.sum(matches >= 0))
                    img_loc = cv2.putText(img=img_loc, text=show_text, org=(30, 30),
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                          thickness=2, lineType=cv2.LINE_AA)

                    img_loc = np.hstack([q_ref_img, img_loc])

                    cv2.imshow('loc', img_loc)

                    key = cv2.waitKey(self.loc_config['show_time'])
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        exit(0)
                continue

            num_inliers = ret['num_inliers']
            inliers = ret['inliers']  # [True, False]

            if gt_qcw is not None:
                q_err, t_err = compute_pose_error(pred_qcw=ret['qvec'],
                                                  pred_tcw=ret['tvec'],
                                                  gt_qcw=gt_qcw,
                                                  gt_tcw=gt_tcw)
            else:
                q_err = 1e2
                t_err = 1e2

            if show:
                img_loc = vis_seg_point(img=q_img, kpts=all_mp2ds,
                                        segs=np.zeros(shape=(all_mp2ds.shape[0],)) + sid, seg_color=seg_color,
                                        radius=9)
                img_loc = vis_inlier(img=img_loc, kpts=all_mp2ds,
                                     inliers=inliers,
                                     radius=9 + 2,
                                     thickness=2)
                img_loc = resize_img(img=img_loc, nh=512)
                show_text = 'order: {:d}/{:d}, k/m/i: {:d}/{:d}/{:d}'.format(
                    i, len(q_loc_segs), q_seg_kpts.shape[0], np.sum(matches >= 0), ret['num_inliers'])
                img_loc = cv2.putText(img=img_loc, text=show_text, org=(30, 30),
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
                show_text = 'r_err:{:.2f}, t_err:{:.2f}'.format(q_err, t_err)
                img_loc = cv2.putText(img=img_loc, text=show_text, org=(30, 80),
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
                img_vis = img_loc

                img_loc_to_show = np.hstack([q_ref_img, img_loc])

                cv2.imshow('loc', img_loc_to_show)
                key = cv2.waitKey(self.loc_config['show_time'])
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(0)

            if save:
                print('save image to ', self.save_dir)
                cv2.imwrite(osp.join(self.save_dir,
                                     q_scene_name.replace('/', '+') + '+' + q_name.replace('/',
                                                                                           '+') + '_order_{:d}_loc.png'.format(
                                         i)), img_loc)

            if ret['num_inliers'] < self.loc_config['min_inliers']:
                print_text = 'qname: {:s} failed due to insufficient {:d} inliers, order {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                    q_name,
                    ret['num_inliers'],
                    i, q_err, t_err)
                print(print_text)
                log_text.append(print_text)

                if ret['num_inliers'] > best_results['num_inliers']:
                    # record the best vrf
                    best_inlier = -1
                    best_vrf_id = -1
                    best_vrf_name = -1
                    all_inliers = np.array(ret['inliers'])
                    for vrf_id in macth_vrf_info.keys():
                        start = macth_vrf_info[vrf_id]['start']
                        end = macth_vrf_info[vrf_id]['end']
                        n_inliers = np.sum(all_inliers[start:end])
                        if n_inliers > best_inlier:
                            best_inlier = n_inliers
                            best_vrf_id = macth_vrf_info[vrf_id]['image_id']
                            best_vrf_name = macth_vrf_info[vrf_id]['image_name']

                    if best_inlier > 0:
                        best_results['ref_img_name'] = best_vrf_name
                        best_results['ref_img_id'] = best_vrf_id
                        best_results['num_inliers'] = ret['num_inliers']
                        best_results['qvec'] = ret['qvec']
                        best_results['tvec'] = ret['tvec']
                        best_results['order'] = i
                        best_results['scene_name'] = pred_scene_name
                        best_results['seg_id'] = sid

                continue
            else:
                loc_success = True

            print_text = 'Find {}/{} 2D-3D inliers'.format(num_inliers, len(inliers))
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

                best_inlier = -1
                best_ref_id = -1
                best_ref_name = -1
                best_scene_name = -1
                all_inliers = np.array(ret['inliers'])
                for vrf_id in macth_vrf_info.keys():
                    start = macth_vrf_info[vrf_id]['start']
                    end = macth_vrf_info[vrf_id]['end']
                    n_inliers = np.sum(all_inliers[start:end])
                    if n_inliers > best_inlier:
                        best_inlier = n_inliers
                        best_ref_id = macth_vrf_info[vrf_id]['image_id']
                        best_ref_name = macth_vrf_info[vrf_id]['image_name']
                        best_scene_name = macth_vrf_info[vrf_id]['scene_name']

                data = {
                    'query_data': {
                        'cfg': cfg,
                        'descriptors': q_descs,
                        'scores': q_scores,
                        'keypoints': q_kpts,
                        'width': width,
                        'height': height,
                        'qvec': ret['qvec'],
                        'tvec': ret['tvec'],
                        'n_inliers': best_inlier,
                        'loc_success': loc_success,
                        'mp2ds': all_mp2ds,
                        'mp3ds': all_mp3ds,
                    },
                    'frame_id': best_ref_id,
                }
                if self.refinement_method == 'matching':
                    ref_ret = pred_sub_map.refine_pose_by_matching(data=data)
                elif self.refinement_method == 'projection':
                    ref_ret = pred_sub_map.refine_pose_by_projection(data=data)

                t_ref = time.time() - t_start

                if ref_ret['success']:
                    if gt_qcw is not None:
                        q_err, t_err = compute_pose_error(pred_qcw=ref_ret['qvec'],
                                                          pred_tcw=ref_ret['tvec'],
                                                          gt_qcw=gt_qcw,
                                                          gt_tcw=gt_tcw)
                    else:
                        q_err, t_err = 1e2, 1e2

                    best_ref_full_name = osp.join(best_scene_name, best_ref_name)
                    print_text = 'Localization of {:s} success with inliers {:d}/{:d} with ref_name: {:s}, order: {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                        q_full_name, ref_ret['num_inliers'], len(ref_ret['inliers']), best_ref_full_name, i, q_err,
                        t_err)
                    print(print_text)
                    log_text.append(print_text)
                    out = {
                        'qvec': ref_ret['qvec'],
                        'tvec': ref_ret['tvec'],
                        'success': True,
                        'log': log_text,
                        'q_err': q_err,
                        't_err': t_err,
                        'time_loc': t_loc,
                        'time_ref': t_ref,
                    }

                    if show:
                        ''' only for refiniement 
                        img_final = vis_inlier(img=q_img, kpts=ref_ret['matched_qkp'], inliers=ref_ret['inliers'],
                                               radius=9 + 2,
                                               thickness=2,
                                               with_outlier=False)
                        img_final = resize_img(img_final, nh=512)
                        show_text = 'order: {:d}/{:d}, k/m/i: {:d}/{:d}/{:d}'.format(
                            i, len(q_loc_segs), q_seg_kpts.shape[0], len(ref_ret['inliers']), ref_ret['num_inliers'])
                        img_final = cv2.putText(img=img_final, text=show_text, org=(30, 30),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                                thickness=2, lineType=cv2.LINE_AA)
                        show_text = 'r_err:{:.2f}, t_err:{:.2f}'.format(q_err, t_err)
                        img_final = cv2.putText(img=img_final, text=show_text, org=(30, 80),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                                thickness=2, lineType=cv2.LINE_AA)

                        img_vis = img_final

                        if save:
                            cv2.imwrite(osp.join(self.save_dir,
                                                 q_scene_name.replace('/', '+') + '+' + q_name.replace('/',
                                                                                                       '+') + '_final.png'),
                                        img_final)

                        cv2.imshow('final', img_final)
                        key = cv2.waitKey(self.loc_config['show_time'])
                        if key == ord('q'):
                            cv2.destroyAllWindows()
                            exit(0)
                        '''

                        vis_out = {
                            'gt_qvec': gt_qcw,
                            'gt_tvec': gt_tcw,
                            'q_err': q_err,
                            't_err': t_err,
                            'query_img': q_img,
                            'img_loc': img_vis,
                            'img_matching': img_loc_matching,
                            'pred_sid': sid,
                            'reference_db_ids': ref_ret['reference_db_ids'],
                            'vrf_image_id': best_ref_id,
                            'start_seg_id': start_seg_id,
                        }

                    return {**out, **vis_out}
                else:
                    continue
            else:
                out = {
                    'gt_qvec': gt_qcw,
                    'gt_tvec': gt_tcw,
                    'qvec': ret['qvec'],
                    'tvec': ret['tvec'],
                    'log': log_text,
                    'n_inliers': len(inliers),
                    'success': True,
                    'q_err': q_err,
                    't_err': t_err,
                    'time_ref': t_ref,
                    'time_loc': t_loc,
                    'img_loc': img_vis,
                    'query_img': q_img,
                    'img_matching': img_loc_matching,

                    'pred_sid': sid,
                    'reference_db_ids': [ref_img_id],
                    'vrf_image_id': ref_img_id,
                    'start_seg_id': start_seg_id,
                }

                return out

        # try to estimate the pose from failed candidates
        if best_results['num_inliers'] >= 4:
            best_pred_scene = best_results['scene_name']
            best_sub_map = self.sub_maps[best_pred_scene]
            print('Try to do localization from best results inliers {},ref_name {}, order {}'.format(
                best_results['num_inliers'],
                best_results['ref_img_name'],
                best_results['order']))
            # if self.loc_config['do_refinement']:
            if self.do_refinement:
                t_start = time.time()
                ref_ret = best_sub_map.refine_pose(data={
                    'query_data': {
                        'cfg': cfg,
                        'descriptors': q_descs,
                        'scores': q_scores,
                        'keypoints': q_kpts,
                        'width': width,
                        'height': height,
                        'qvec': None,
                        'tvec': None,
                        'n_inliers': best_results['num_inliers'],
                        'loc_success': False,
                        'mp2ds': None,
                        'mp3ds': None,
                    },
                    'frame_id': best_results['ref_img_id'],
                })

                t_ref = time.time() - t_start

                if ref_ret['success']:
                    if gt_sub_map.gt_poses is not None and q_name in gt_sub_map.gt_poses.keys():
                        q_err, t_err = compute_pose_error(pred_qcw=ref_ret['qvec'],
                                                          pred_tcw=ref_ret['tvec'],
                                                          gt_qcw=gt_sub_map.gt_poses[q_name]['qvec'],
                                                          gt_tcw=gt_sub_map.gt_poses[q_name]['tvec'])
                    else:
                        q_err, t_err = 1e2, 1e2
                    best_ref_full_name = osp.join(best_pred_scene, best_results['ref_img_name'])
                    print_text = 'Localization of {:s} success with inliers {:d}/{:d} ref_name: {:s}, order: {:d}, q_err: {:.2f}, t_err: {:.2f}'.format(
                        q_full_name,
                        ref_ret['num_inliers'], len(ref_ret['inliers']),
                        best_ref_full_name, best_results['order'], q_err, t_err)
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

                    if show:
                        '''
                        img_final = vis_inlier(img=q_img, kpts=ref_ret['matched_qkp'], inliers=ref_ret['inliers'],
                                               radius=9 + 2,
                                               thickness=2)
                        img_final = resize_img(img_final, nh=512)
                        show_text = 'order: {:d}/{:d}, k/m/i: {:d}/{:d}/{:d}'.format(
                            best_results['order'], len(q_loc_segs), q_seg_kpts.shape[0], len(ref_ret['inliers']),
                            ref_ret['num_inliers'])
                        img_final = cv2.putText(img=img_final, text=show_text, org=(30, 30),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                                thickness=2, lineType=cv2.LINE_AA)
                        show_text = 'r_err:{:.2f}, t_err:{:.2f}'.format(q_err, t_err)
                        img_final = cv2.putText(img=img_final, text=show_text, org=(30, 80),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                                thickness=2, lineType=cv2.LINE_AA)

                        img_vis = img_final

                        cv2.imshow('final', img_final)
                        key = cv2.waitKey(self.loc_config['show_time'])

                        if save:
                            cv2.imwrite(osp.join(self.save_dir,
                                                 q_scene_name.replace('/', '+') + '+' + q_name.replace('/',
                                                                                                       '+') + '_final.png'),
                                        img_final)
                        if key == ord('q'):
                            cv2.destroyAllWindows()
                            exit(0)
                        '''

                    vis_out = {
                        'gt_qvec': gt_qcw,
                        'gt_tvec': gt_tcw,

                        # 'time_loc': t_loc,
                        # 'time_ref': t_ref,
                        'img_loc': img_vis,
                        'query_img': q_img,
                        'img_matching': img_loc_matching,
                        'pred_sid': sid,
                        'reference_db_ids': ref_ret['reference_db_ids'],
                        'vrf_image_id': best_results['ref_img_id'],
                        'start_seg_id': start_seg_id,
                    }
                    return {**out, **vis_out}

            print_text = 'qname: {:s} find the best among all candidates'.format(q_name)
            log_text.append(print_text)

            if gt_sub_map.gt_poses is not None and q_name in gt_sub_map.gt_poses.keys():
                q_err, t_err = compute_pose_error(pred_qcw=best_results['qvec'],
                                                  pred_tcw=best_results['tvec'],
                                                  gt_qcw=gt_sub_map.gt_poses[q_name]['qvec'],
                                                  gt_tcw=gt_sub_map.gt_poses[q_name]['tvec'])
            else:
                q_err, t_err = 1e2, 1e2

            out = {
                'gt_qvec': gt_qcw,
                'gt_tvec': gt_tcw,
                'qvec': best_results['qvec'],
                'tvec': best_results['tvec'],
                'success': False,
                'log': log_text,
                'q_err': q_err,
                't_err': t_err,

                'time_loc': t_loc,
                'time_ref': t_ref,
                'img_loc': img_vis,
                'query_img': q_img,

                'img_matching': img_loc_matching,
                'pred_sid': sid,
                'reference_db_ids': [ref_img_id],
                'vrf_image_id': best_results['ref_img_id'],
                'start_seg_id': start_seg_id,

            }
            return out

        best_scene_name = self.sid_scene_name[query_sids[0]]
        best_sub_map = self.sub_maps[best_scene_name]
        best_sid_in_sub_scene = query_sids[0] - self.scene_name_start_sid[best_scene_name]
        best_ref_img_id = best_sub_map.seg_vrf[best_sid_in_sub_scene][0]['image_id']
        best_ref_name = best_sub_map.seg_vrf[best_sid_in_sub_scene][0]['image_name']
        qvec = best_sub_map.map_images[best_ref_img_id].qvec
        tvec = best_sub_map.map_images[best_ref_img_id].tvec
        if gt_sub_map.gt_poses is not None and q_name in gt_sub_map.gt_poses.keys():
            q_err, t_err = compute_pose_error(pred_qcw=qvec,
                                              pred_tcw=tvec,
                                              gt_qcw=gt_sub_map.gt_poses[q_name]['qvec'],
                                              gt_tcw=gt_sub_map.gt_poses[q_name]['tvec'])
        else:
            q_err, t_err = 1e2, 1e2
        best_ref_full_name = osp.join(best_scene_name, best_ref_name)
        print_text = 'Localization of {:s} failed with best reference image {:s} as results, q_err: {:.2f}, t_err: {:.2f}'.format(
            q_full_name, best_ref_full_name, q_err, t_err)
        print(print_text)
        log_text.append(print_text)
        return {
            'gt_qvec': gt_qcw,
            'gt_tvec': gt_tcw,
            'qvec': qvec,
            'tvec': tvec,
            'q_err': q_err,
            't_err': t_err,
            'success': False,
            'log': log_text,
            'time_loc': t_loc,
            'time_ref': t_ref,
            'img_loc': img_loc,
            'query_img': q_img,

            'img_matching': img_loc_matching,
            'pred_sid': sid,
            'reference_db_ids': [ref_img_id],
            'vrf_image_id': best_results['ref_img_id'],
            'start_seg_id': start_seg_id,

        }

    def retrieve_vrf_by_recognition(self, pred_sids: dict, mode: int = 1):
        query_sids = pred_sids.keys()
        overlap_sids = {}
        for sid in query_sids:
            if sid in overlap_sids.keys():
                continue
            pred_scene_name = self.sid_scene_name[sid]
            pred_sub_map = self.sub_maps[pred_scene_name]
            pred_sid_in_sub_scene = sid - self.scene_name_start_sid[pred_scene_name]
            pred_vrf = pred_sub_map.seg_vrf[pred_sid_in_sub_scene]
            org_p3ds = pred_vrf['original_points3d'].tolist()
            # print('org_p3ds: ', org_p3ds)
            org_sids = []
            for pid in org_p3ds:
                if pid not in pred_sub_map.map_seg.keys():
                    continue
                p3d_sid = pred_sub_map.map_seg[pid] + sid
                if p3d_sid in org_sids:
                    continue
                org_sids.append(p3d_sid)

            overlaps = [v for v in query_sids if v in org_sids]
            overlap_sids[sid] = overlaps

        if mode == 0:
            out = {k: v for k, v in sorted(overlap_sids.items(), key=lambda item: 0 - len(item[1]))}
        elif mode == 1:
            overlap_nkpts = {}
            for k in overlap_sids.keys():
                n = 0
                for c in overlap_sids[k]:
                    n += pred_sids[c][0]
                overlap_nkpts[k] = n
            out = {k: v for k, v in sorted(overlap_nkpts.items(), key=lambda item: 0 - item[1])}
        elif mode == 2:  # score
            overlap_score = {}
            for k in overlap_sids.keys():
                s = 0
                for c in overlap_sids[k]:
                    s += pred_sids[c][1]
                overlap_score[k] = s
            out = {k: v for k, v in sorted(overlap_score.items(), key=lambda item: 0 - item[1])}

        return out

    def build_matches_between_query_and_vrf(self, query_matching_data, sid, pred_sid_in_sub_scene, pred_sub_map,
                                            pred_vrf,
                                            seg_wise_matching=False,
                                            **kwargs):
        if self.loc_config['with_original']:
            org_p3d_ids = pred_vrf['original_points3d']
            if seg_wise_matching:
                p3d_ids = []
                for v in org_p3d_ids:
                    if v in pred_sub_map.map_seg.keys():
                        if pred_sub_map.map_seg[v] == pred_sid_in_sub_scene:
                            p3d_ids.append(v)
                if len(p3d_ids) < self.loc_config['min_kpts']:
                    p3d_ids = org_p3d_ids
            else:
                p3d_ids = org_p3d_ids

            p3d_out = pred_sub_map.get_p3ds_by_ids(p3d_ids=p3d_ids)
            p3d_descs, p3ds, p3ds_errors, valid_p3d_ids = p3d_out['descriptors'], p3d_out['xyzs'], p3d_out['errors'], \
                p3d_out['valid_p3d_ids']
            p3d_segs = []
            for v in valid_p3d_ids:
                if v in pred_sub_map.map_seg.keys():
                    p3d_segs.append(pred_sub_map.map_seg[v])
                else:
                    p3d_segs.append(-1)
            p3d_segs = np.array(p3d_segs)
        else:
            # p3d_ids = pred_sub_map.seg_map[pred_sid_in_sub_scene]
            org_p3d_ids = pred_vrf['original_points3d']
            if seg_wise_matching:
                p3d_ids = []
                for v in org_p3d_ids:
                    if v in pred_sub_map.map_seg.keys():
                        if pred_sub_map.map_seg[v] == pred_sid_in_sub_scene:
                            p3d_ids.append(v)
                if len(p3d_ids) < self.loc_config['min_kpts']:
                    p3d_ids = org_p3d_ids
            else:
                p3d_ids = org_p3d_ids

            p3d_out = pred_sub_map.get_p3ds_by_ids(p3d_ids=p3d_ids)
            p3d_descs, p3ds, p3ds_errors = p3d_out['descriptors'], p3d_out['xyzs'], p3d_out['errors']
            p3d_segs = np.array([sid for _ in range(p3ds.shape[0])])
            valid_p3d_ids = p3d_out['valid_p3d_ids']

        map_matching_data = {
            'descriptors': p3d_descs,
            'scores': 1. / np.clip(p3ds_errors * 5, a_min=1.0, a_max=20),
            'xyzs': p3ds,
            'camera': pred_vrf['camera'],
            'qvec': pred_vrf['qvec'],
            'tvec': pred_vrf['tvec'],
            'p3d_ids': valid_p3d_ids,
        }

        with torch.no_grad():
            match_out = pred_sub_map.match(
                query_data=query_matching_data,
                map_data=map_matching_data,
            )

        match_out['p3ds'] = p3ds
        match_out['p3d_segs'] = p3d_segs
        match_out['p3d_ids'] = p3d_out['valid_p3d_ids']

        return match_out
