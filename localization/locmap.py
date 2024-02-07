# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> locmap
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 15:20
=================================================='''
import torch
from collections import defaultdict
import cv2
import os
import os.path as osp
import numpy as np
import pycolmap
import yaml
import time
from copy import deepcopy
from loc.utils.tools import compute_pose_error
from colmap_utils.camera_intrinsics import intrinsics_from_camera
from colmap_utils.read_write_model import qvec2rotmat, read_model, read_compressed_model, intrinsics_from_camera
from recognition.vis_seg import vis_seg_point, generate_color_dic, vis_inlier
from tools.common import resize_img
from recognition.vis_seg import plot_matches


class SingleLocMap:
    def __init__(self, config, matcher, with_compress=False):
        self.config = config
        self.image_path_prefix = self.config['image_path_prefix']
        if not with_compress:
            self.map_cameras, self.map_images, self.map_p3ds = read_model(
                path=osp.join(config['segment_path'], 'model'), ext='.bin')
            self.map_desc = np.load(osp.join(config['segment_path'], 'point3D_desc.npy'),
                                    allow_pickle=True)[()]
        else:
            self.map_cameras, self.map_images, self.map_p3ds = read_compressed_model(
                path=osp.join(config['segment_path'], 'compress_model_{:s}'.format(config['cluster_method'])),
                ext='.bin')
            self.map_desc = np.load(osp.join(config['segment_path'], 'compress_model_{:s}/point3D_desc.npy'.format(
                config['cluster_method'])), allow_pickle=True)[()]

        print('Load {} cameras {} images {} 3D points'.format(len(self.map_cameras), len(self.map_images),
                                                              len(self.map_p3ds)))

        seg_data = np.load(osp.join(config['segment_path'],
                                    'point3D_cluster_n{:d}_{:s}_{:s}.npy'.format(config['n_cluster'],
                                                                                 config['cluster_mode'],
                                                                                 config['cluster_method'])),
                           allow_pickle=True)[()]
        p3d_id = seg_data['id']
        seg_id = seg_data['label']
        self.map_seg = {p3d_id[i]: seg_id[i] for i in range(p3d_id.shape[0])}
        self.seg_map = {}
        for k in self.map_seg.keys():
            sid = self.map_seg[k]
            if sid in self.seg_map.keys():
                self.seg_map[sid].append(k)
            else:
                self.seg_map[sid] = [k]

        print('Load {} segments and {} 3d points'.format(len(self.seg_map.keys()), len(self.map_seg.keys())))

        # Load vrf data
        self.seg_vrf = np.load(osp.join(config['segment_path'],
                                        'point3D_vrf_n{:d}_{:s}_{:s}.npy'.format(config['n_cluster'],
                                                                                 config['cluster_mode'],
                                                                                 config['cluster_method'])),
                               allow_pickle=True)[()]
        self.ignore_index = -1

        # load sc data
        if osp.isfile(osp.join(config['segment_path'], 'sc_mean_scale.txt')):
            with open(osp.join(config['segment_path'], 'sc_mean_scale.txt'), 'r') as f:
                l = f.readline()
                l = l.strip().split()
                mean = np.array([float(v) for v in l[0:3]])
                scale = np.array([float(v) for v in l[3:]])
                self.sc_data = {
                    'mean_xyz': mean,
                    'scale_xyz': scale,
                }
                print('sc mean {}, scale {}'.format(mean, scale))

        self.gt_poses = {}
        if config['gt_pose_path'] is not None:
            gt_pose_path = osp.join(config['dataset_path'], config['gt_pose_path'])
            self.read_gt_pose(path=gt_pose_path)

        self.matcher = matcher

    def get_covisible_frames(self, data):
        frame_id = data['frame_id']
        n_frame = self.config['localization']['covisibility_frame']
        observed = self.map_images[frame_id].point3D_ids
        covis = defaultdict(int)
        for pid in observed:
            if pid == -1:
                continue
            if pid not in self.map_p3ds.keys():
                continue
            for img_id in self.map_p3ds[pid].image_ids:
                # if img_id != frame_id:
                covis[img_id] += 1
        # print('Find {:d} valid connected frames'.format(len(covis.keys())))

        covis_ids = np.array(list(covis.keys()))
        covis_num = np.array([covis[i] for i in covis_ids])

        if len(covis_ids) <= n_frame:
            sel_covis_ids = covis_ids[np.argsort(-covis_num)]
        else:
            ind_top = np.argpartition(covis_num, -n_frame)
            ind_top = ind_top[-n_frame:]  # unsorted top k
            ind_top = ind_top[np.argsort(-covis_num[ind_top])]
            sel_covis_ids = [covis_ids[i] for i in ind_top]

        # print('Retain {:d} valid connected frames'.format(len(sel_covis_ids)))

        return sel_covis_ids

    def get_covisible_frames_by_frame_id(self, data):
        frame_id = data['frame_id']
        n_frame = self.config['localization']['covisibility_frame']
        sel_covis_ids = [frame_id]
        offset = 5
        for fi in range(-n_frame // 2, n_frame // 2 + 1):
            new_frame_id = frame_id + fi * offset
            if new_frame_id not in self.map_images.keys():
                continue
            if new_frame_id in sel_covis_ids:
                continue
            sel_covis_ids.append(new_frame_id)

        # print('Retain {:d} valid connected frames'.format(len(sel_covis_ids)))

        return sel_covis_ids

    def outlier_filtering_by_projection(self, Tcw, K, width, height, xyzs, min_depth=0, max_depth=50):
        xyzs_homo = np.hstack([xyzs, np.ones((xyzs.shape[0], 1), dtype=xyzs.dtype)])
        proj_uvs = K @ (Tcw @ xyzs_homo.transpose())
        proj_uvs = proj_uvs.transpose()  # [N, 3]
        proj_uvs[:, 0] = proj_uvs[:, 0] / proj_uvs[:, 2]
        proj_uvs[:, 1] = proj_uvs[:, 1] / proj_uvs[:, 2]
        mask = (proj_uvs[:, 0] >= 0) * (proj_uvs[:, 0] < width) * (proj_uvs[:, 1] >= 0) * (proj_uvs[:, 1] < height)
        mask = mask * (proj_uvs[:, 2] > min_depth) * (proj_uvs[:, 2] < max_depth)
        return mask

    def refine_pose(self, data, remove_overlap=True):
        query_data = data['query_data']
        query_cfg = data['query_data']['cfg']
        n_init_inliers = data['query_data']['n_inliers']
        width = query_cfg['width']
        height = query_cfg['height']
        K = intrinsics_from_camera(camera_model=query_cfg['model'], params=query_cfg['params'])
        loc_success = data['query_data']['loc_success']

        frame_id = data['frame_id']
        db_ids = None

        t_start = time.time()
        for sid in self.seg_vrf.keys():
            if 0 not in self.seg_vrf[sid].keys():
                continue
            if self.seg_vrf[sid][0]['image_id'] == frame_id:
                if 'covisible_frame_ids' in self.seg_vrf[sid][0].keys():
                    db_ids = self.seg_vrf[sid][0]['covisible_frame_ids']
                    n_frame = self.config['localization']['covisibility_frame']
                    if len(db_ids) > n_frame:
                        db_ids = [db_ids[i] for i in range(n_frame)]
                    break
        if db_ids is not None:
            valid_db_ids = [v for v in db_ids if v in self.map_images.keys()]
            db_ids = valid_db_ids
            print('Find {} covisible frames from vrf'.format(len(db_ids)))
        else:
            db_ids = self.get_covisible_frames(data=data)
            db_ids = list(db_ids)

        # print('db_ids: ', type(db_ids), type(frame_id))
        if loc_success and frame_id in db_ids:
            init_mp2ds = data['query_data']['mp2ds']
            init_mp3ds = data['query_data']['mp3ds']
            db_ids.remove(frame_id)
        else:
            init_mp2ds = None
            init_mp3ds = None

        n_init_frames = len(db_ids)
        n_max_frames = self.config['localization']['covisibility_frame']
        n_min_frames = 5
        n_max_inliers = 1024
        n_min_inliers = 64
        if n_init_inliers >= n_max_inliers:
            n_valid_frames = n_min_frames
        elif n_init_inliers < n_min_inliers:
            n_valid_frames = n_max_frames
        else:
            n_valid_frames = (n_max_inliers - n_init_inliers) / (n_max_inliers - n_min_inliers) * (
                    n_max_frames - n_min_frames) + n_min_frames
            n_valid_frames = int(np.ceil(n_valid_frames))
        db_ids = db_ids[:n_valid_frames]
        # print('Valid frames: {:d} - {:d}-{:d}'.format(n_init_inliers, len(db_ids), n_init_frames))
        # print('Time of getting covisible frames: {:.2f}s'.format(time.time() - t_start))

        qkps = query_data['keypoints']

        matched_xyzs = []
        matched_qkp = []
        matched_p3d_ids = []

        all_q_kpq_p3d_matches = {}
        n_total_matches = 0 if init_mp2ds is None else init_mp2ds.shape[0]

        t_start = time.time()
        for idx, db_id in enumerate(db_ids):
            p3d_ids = [v for v in self.map_images[db_id].point3D_ids if v >= 0]
            p3d_data = self.get_p3ds_by_ids(p3d_ids=p3d_ids)
            descs = p3d_data['descriptors']
            xyzs = p3d_data['xyzs']
            errors = p3d_data['errors']

            db_camera = self.map_cameras[self.map_images[db_id].camera_id]
            map_matching_data = {
                'descriptors': descs,
                'xyzs': xyzs,
                'scores': 1. / np.clip(errors * 5, a_min=1, a_max=20.),
                'camera': {
                    'model': db_camera.model,
                    'width': db_camera.width,
                    'height': db_camera.height,
                    'params': db_camera.params,
                },
                'qvec': self.map_images[db_id].qvec,
                'tvec': self.map_images[db_id].tvec,
                'p3d_ids': p3d_ids,
            }

            match_out = self.match(
                query_data=query_data,
                map_data=map_matching_data,
            )

            matches = match_out['matches']
            for mi in range(matches.shape[0]):
                if matches[mi] < 0:
                    continue

                if remove_overlap:
                    if mi in all_q_kpq_p3d_matches.keys():
                        if p3d_ids[matches[mi]] in all_q_kpq_p3d_matches[mi]:
                            continue
                        else:
                            all_q_kpq_p3d_matches[mi].append(p3d_ids[matches[mi]])
                    else:
                        all_q_kpq_p3d_matches[mi] = [p3d_ids[matches[mi]]]

                matched_qkp.append(qkps[mi])
                matched_xyzs.append(xyzs[matches[mi]])
                matched_p3d_ids.append(p3d_ids[matches[mi]])
                n_total_matches = len(matched_qkp)

            if self.config['localization']['refinement_max_matches'] > 0:
                if n_total_matches >= self.config['localization']['refinement_max_matches']:
                    break

        print('Time of matching: {:.2f}s'.format(time.time() - t_start))

        matched_qkp = np.array(matched_qkp, float).reshape(-1, 2) + 0.5
        matched_xyzs = np.array(matched_xyzs, float).reshape(-1, 3)

        if init_mp2ds is not None and init_mp3ds is not None:
            matched_qkp = np.vstack([matched_qkp, init_mp2ds])
            matched_xyzs = np.vstack([matched_xyzs, init_mp3ds])
            print('Add initial matches - {:d}-{:d}'.format(init_mp2ds.shape[0], matched_qkp.shape[0]))

        print_text = 'Refinement by matching. Get {:d} covisible frames with {:d} matches for optimization'.format(
            len(db_ids), matched_xyzs.shape[0])
        print(print_text)

        t_start = time.time()
        ret = pycolmap.absolute_pose_estimation(matched_qkp, matched_xyzs, query_cfg,
                                                max_error_px=self.config['localization']['threshold'],
                                                min_num_trials=1000, max_num_trials=10000, confidence=0.995)
        print('Time of RANSAC: {:.2f}s'.format(time.time() - t_start))

        ret['matched_qkp'] = matched_qkp
        ret['reference_db_ids'] = db_ids

        return ret

    def find_matches_by_projection(self, qkps, qdescs, map_xyzs, map_descs, map_pids, Tcw, K, width, height):
        map_xyzs_homo = np.hstack([map_xyzs, np.ones((map_xyzs.shape[0], 1), dtype=map_xyzs.dtype)])
        map_descs = np.array(map_descs)  # [N x D]
        proj_3d = K @ (Tcw @ map_xyzs_homo.transpose())  # [3 x N]
        proj_3d = proj_3d.transpose()
        proj_uv = proj_3d[:, :2]
        proj_depth = proj_3d[:, 2]
        proj_uv[:, 0] = proj_uv[:, 0] / proj_depth
        proj_uv[:, 1] = proj_uv[:, 1] / proj_depth
        mask = (proj_depth > 0) * (proj_uv[:, 0] >= self.config['localization']['refinement_radius']) * (
                proj_uv[:, 0] < width - self.config['localization']['refinement_radius']) * (
                       proj_uv[:, 1] >= self.config['localization']['refinement_radius']) * (
                       proj_uv[:, 1] < height - self.config['localization']['refinement_radius'])
        valid_xyzs = map_xyzs[mask]
        valid_p3d_ids = map_pids[mask]
        valid_proj_uv = proj_uv[mask]
        valid_descs = map_descs[mask]

        if valid_xyzs.shape[0] <= 1:
            return np.array([]), np.array([]), np.array([])

        with torch.no_grad():
            qdescs_cuda = torch.from_numpy(qdescs).cuda()
            valid_descs_cuda = torch.from_numpy(valid_descs).cuda()
            qkps_cuda = torch.from_numpy(qkps).cuda()
            valid_proj_uv_cuda = torch.from_numpy(valid_proj_uv).cuda()

            proj_dist = qkps_cuda.unsqueeze(-1) - valid_proj_uv_cuda.t().unsqueeze(0)
            proj_dist = torch.sqrt((proj_dist ** 2).sum(dim=1))  # [N, M]
            desc_dist = 2 - 2 * qdescs_cuda.float() @ valid_descs_cuda.t().float()  # []
            desc_dist = torch.sqrt(desc_dist + 0.000001)

            proj_dist_mask_outlier = (proj_dist > self.config['localization']['refinement_radius'])
            proj_mask = torch.zeros_like(proj_dist)
            proj_mask[proj_dist_mask_outlier] = 10
            total_dist = desc_dist + proj_mask
            # print('dist: ', proj_mask.shape, total_dist.shape)

            dist_values, dist_ids = torch.topk(total_dist, k=2, dim=1, largest=False)
            iids = torch.arange(qkps.shape[0])
            desc_dist0 = desc_dist[iids, dist_ids[:, 0]].cpu().numpy()
            desc_dist1 = desc_dist[iids, dist_ids[:, 1]].cpu().numpy()
            mask = (desc_dist1 < 10) * (desc_dist0 < desc_dist1 * self.config['localization']['refinement_nn_ratio'])
            nn_ids = dist_ids[:, 0].cpu().numpy()
            matched_qkps = qkps[mask]
            matched_xyzs = valid_xyzs[nn_ids[mask]]
            matched_p3d_ids = valid_p3d_ids[nn_ids[mask]]

        return matched_qkps, matched_xyzs, matched_p3d_ids

    def refine_pose_by_projection(self, data):
        query_data = data['query_data']
        qkps = query_data['keypoints']
        qdescs = query_data['descriptors']
        query_cfg = data['query_data']['cfg']
        init_qcw = query_data['qvec']
        init_tcw = query_data['tvec']
        init_Rcw = qvec2rotmat(qvec=init_qcw)
        Tcw = np.hstack([init_Rcw, init_tcw.reshape(3, 1)])  # [3, 4]
        camera_model = query_cfg['model']
        params = query_cfg['params']
        K = intrinsics_from_camera(camera_model=camera_model, params=params)  # [3 x 3]
        width = query_cfg['width']
        height = query_cfg['height']

        frame_id = data['frame_id']
        db_ids = None
        for sid in self.seg_vrf.keys():
            if 0 not in self.seg_vrf[sid].keys():
                continue
            if self.seg_vrf[sid][0]['image_id'] == frame_id:
                if 'covisible_frame_ids' in self.seg_vrf[sid][0].keys():
                    db_ids = self.seg_vrf[sid][0]['covisible_frame_ids']
                    n_frame = self.config['localization']['covisibility_frame']
                    if len(db_ids) > n_frame:
                        db_ids = [db_ids[i] for i in range(n_frame)]
                    break
        if db_ids is not None:
            valid_db_ids = [v for v in db_ids if v in self.map_images.keys()]
            db_ids = valid_db_ids
            print('Find {} covisible frames from vrf'.format(len(db_ids)))
        else:
            db_ids = self.get_covisible_frames(data=data)

        all_matched_qkps = []
        all_matched_xyzs = []
        all_matched_pids = []
        n_total = 0
        for idx, db_id in enumerate(db_ids):
            map_p3d_ids = []
            map_xyzs = []
            map_descriptors = []
            for pid in self.map_images[db_id].point3D_ids:
                if pid not in self.map_p3ds.keys():
                    continue
                if pid in map_p3d_ids:
                    continue
                if pid in all_matched_pids:  # save time
                    continue
                map_p3d_ids.append(pid)
                map_xyzs.append(self.map_p3ds[pid].xyz)
                map_descriptors.append(self.map_desc[pid])

            if len(map_xyzs) <= 1:
                continue

            map_xyzs = np.array(map_xyzs)
            map_descriptors = np.array(map_descriptors)
            map_p3d_ids = np.array(map_p3d_ids)

            matched_qkps, matched_xyzs, matched_pids = self.find_matches_by_projection(qkps=qkps, qdescs=qdescs,
                                                                                       map_xyzs=map_xyzs,
                                                                                       map_descs=map_descriptors,
                                                                                       map_pids=map_p3d_ids,
                                                                                       Tcw=Tcw, K=K, width=width,
                                                                                       height=height)

            if matched_qkps.shape[0] > 0:
                all_matched_qkps.append(matched_qkps)
                all_matched_xyzs.append(matched_xyzs)
                all_matched_pids.extend(list(matched_pids))

                n_total += matched_qkps.shape[0]

            if self.config['localization']['refinement_max_matches'] > 0:
                if n_total >= self.config['localization']['refinement_max_matches']:
                    break

        all_matched_qkps = np.vstack(all_matched_qkps)
        all_matched_xyzs = np.vstack(all_matched_xyzs)

        print_text = 'Refine by projection. Get {:d} covisible frames with {:d} matches for optimization'.format(
            len(db_ids), all_matched_xyzs.shape[0])
        print(print_text)

        inlier_mask = np.array([True for v in range(all_matched_qkps.shape[0])])
        ret = pycolmap.pose_refinement(init_tcw, init_qcw, all_matched_qkps, all_matched_xyzs, inlier_mask, query_cfg)
        ret['inliers'] = inlier_mask.tolist()
        ret['num_inliers'] = matched_qkps.shape[0]
        ret['matched_qkp'] = all_matched_qkps
        ret['reference_db_ids'] = db_ids
        return ret

    def read_query_info(self, query_fn, name_prefix=''):
        self.query_info = read_query_info(query_fn=query_fn)

    def read_virtual_referenece_frame(self, path):
        data = np.load(path, allow_pickle=True)[()]
        self.seg_vrf = data

    def generate_virtual_reference_frame(self, p3ds, qvec, tvec, cam_info, p3d_ids):
        model = cam_info['model']
        params = cam_info['params']
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = params[0]
            cx = params[1]
            cy = params[2]
        elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
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

        height = cam_info['height']
        width = cam_info['width']
        Rcw = qvec2rotmat(qvec=qvec)
        tcw = tvec.reshape(3, )
        Tcw = np.eye(4, dtype=float)
        Tcw[:3, :3] = Rcw
        Tcw[:3, 3] = tcw

        p3ds_homo = np.hstack([p3ds, np.ones(shape=(p3ds.shape[0], 1), dtype=float)])
        proj_p3ds = K @ ((Tcw @ p3ds_homo.transpose())[:3, :])
        proj_p3ds = proj_p3ds.transpose()  # [N, 3]
        proj_p3ds[:, 0] = proj_p3ds[:, 0] / proj_p3ds[:, 2]
        proj_p3ds[:, 1] = proj_p3ds[:, 1] / proj_p3ds[:, 2]

        mask_in_image = (proj_p3ds[:, 0] >= 0) & (proj_p3ds[:, 0] < width) & (proj_p3ds[:, 1] >= 0) & (
                proj_p3ds[:, 1] < height)
        # mask_depth = (proj_p3ds[:, 2] > 0) & (proj_p3ds[:, 2] <= 80)
        mask_depth = proj_p3ds[:, 2] > 0
        mask = mask_in_image * mask_depth

        return {
            'keypoints': proj_p3ds[:, :2].astype(float),
            'mask': mask,
        }

    def get_p3ds_by_ids(self, p3d_ids, min_obs=-1):
        descs = []
        xyzs = []
        errors = []
        valid_p3d_ids = []
        for pid in p3d_ids:
            if pid not in self.map_p3ds.keys():
                continue
            if pid not in self.map_desc.keys():
                continue
            obs = len(self.map_p3ds[pid].image_ids)
            if obs < min_obs:
                continue
            xyzs.append(self.map_p3ds[pid].xyz)
            errors.append(self.map_p3ds[pid].error)
            descs.append(self.map_desc[pid].transpose())
            valid_p3d_ids.append(pid)

        return {
            'descriptors': np.array(descs),
            'xyzs': np.array(xyzs),
            'errors': np.array(errors),
            'valid_p3d_ids': valid_p3d_ids,
        }

    def read_gt_pose(self, path, name_prefix=''):
        self.gt_poses = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split()
                # self.gt_poses[l[0].split('/')[-1]] = {  # make aachen and robotcar work
                self.gt_poses[name_prefix + l[0]] = {
                    'qvec': np.array([float(v) for v in l[1:5]], float),
                    'tvec': np.array([float(v) for v in l[5:]], float),
                }

        print('Load {} gt poses'.format(len(self.gt_poses.keys())))

    def match(self, query_data, map_data, in_plane=True):
        return self.match_by_vrf(query_data=query_data, map_data=map_data, in_plane=in_plane)

    def match_by_vrf(self, query_data, map_data, in_plane=True, use_nn=False):
        q_descs = query_data['descriptors']
        q_scores = query_data['scores']
        q_keypoints = query_data['keypoints']
        q_height = query_data['height']
        q_width = query_data['width']

        map_descs = map_data['descriptors']
        map_scores = map_data['scores']  # can be from projection error
        map_p3ds = map_data['xyzs']
        map_p3d_ids = map_data['p3d_ids']
        map_height = map_data['camera']['height']
        map_width = map_data['camera']['width']

        ref_out = self.generate_virtual_reference_frame(p3ds=map_p3ds,
                                                        qvec=map_data['qvec'],
                                                        tvec=map_data['tvec'],
                                                        cam_info=map_data['camera'],
                                                        p3d_ids=map_p3d_ids, )
        ref_kpts = ref_out['keypoints']
        ref_mask = ref_out['mask']

        if in_plane:
            matches = np.zeros(shape=(q_keypoints.shape[0],), dtype=int) - 1  #
            map_ids = np.where(ref_mask)[0]

            if map_ids.shape[0] > 0:
                indices0 = self.matcher({
                    'descriptors0': torch.from_numpy(q_descs)[None].permute(0, 2, 1).cuda().float(),
                    'keypoints0': torch.from_numpy(q_keypoints)[None].cuda().float(),
                    'scores0': torch.from_numpy(q_scores)[None].cuda().float(),
                    'image_shape0': (1, 3, q_height, q_width),

                    'descriptors1': torch.from_numpy(map_descs[ref_mask])[None].permute(0, 2, 1).cuda().float(),
                    'keypoints1': torch.from_numpy(ref_kpts[ref_mask])[None].cuda().float(),
                    'scores1': torch.from_numpy(map_scores[ref_mask])[None].cuda().float(),
                    'image_shape1': (1, 3, map_height, map_width),
                })
                # print('indices0: ', indices0.shape, np.sum(indices0 >= 0))
                for mi in range(indices0.shape[0]):
                    if indices0[mi] >= 0:
                        matches[mi] = map_ids[indices0[mi]]
        else:
            indices0 = self.matcher({
                'descriptors0': torch.from_numpy(q_descs)[None].permute(0, 2, 1).cuda().float(),
                'keypoints0': torch.from_numpy(q_keypoints)[None].cuda().float(),
                'scores0': torch.from_numpy(q_scores)[None].cuda().float(),
                'image_shape0': (1, 3, q_height, q_width),

                'descriptors1': torch.from_numpy(map_descs)[None].permute(0, 2, 1).cuda().float(),
                'keypoints1': torch.from_numpy(ref_kpts)[None].cuda().float(),
                'scores1': torch.from_numpy(map_scores)[None].cuda().float(),
                'image_shape1': (1, 3, map_height, map_width),

            })
            matches = indices0

        return {
            'matches': matches,
            'ref_keypoints': ref_kpts,
            'ref_mask': ref_mask,
        }
