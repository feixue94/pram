# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> map3d
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 10:25
=================================================='''
import numpy as np
from collections import defaultdict
import os.path as osp
import pycolmap
import logging
import time

import torch

from localization.refframe import RefFrame
from localization.frame import Frame
from localization.point3d import Point3D
from colmap_utils.read_write_model import qvec2rotmat, read_model, read_compressed_model
from localization.utils import read_gt_pose


class SingleMap3D:
    def __init__(self, config, matcher, with_compress=False, start_sid: int = 0):
        self.config = config
        self.matcher = matcher
        self.image_path_prefix = self.config['image_path_prefix']
        self.start_sid = start_sid  # for a dataset with multiple scenes
        if not with_compress:
            cameras, images, p3ds = read_model(
                path=osp.join(config['landmark_path'], 'model'), ext='.bin')
            p3d_descs = np.load(osp.join(config['landmark_path'], 'point3D_desc.npy'),
                                allow_pickle=True)[()]
        else:
            cameras, images, p3ds = read_compressed_model(
                path=osp.join(config['landmark_path'], 'compress_model_{:s}'.format(config['cluster_method'])),
                ext='.bin')
            p3d_descs = np.load(osp.join(config['landmark_path'], 'compress_model_{:s}/point3D_desc.npy'.format(
                config['cluster_method'])), allow_pickle=True)[()]

        print(
            'Load {} cameras {} images {} 3D points from compress {}'.format(len(cameras), len(images), len(p3d_descs),
                                                                             with_compress))

        seg_data = np.load(
            osp.join(config['landmark_path'], 'point3D_cluster_n{:d}_{:s}_{:s}.npy'.format(config['n_cluster'],
                                                                                           config['cluster_mode'],
                                                                                           config['cluster_method'])),
            allow_pickle=True)[()]

        p3d_id = seg_data['id']
        seg_id = seg_data['label']
        p3d_seg = {p3d_id[i]: seg_id[i] for i in range(p3d_id.shape[0])}
        seg_p3d = {}
        for k in p3d_seg.keys():
            sid = p3d_seg[k]
            if sid in seg_p3d.keys():
                seg_p3d[sid].append(k)
            else:
                seg_p3d[sid] = [k]

        print('Load {} segments and {} 3d points'.format(len(seg_p3d.keys()), len(p3d_seg.keys())))
        seg_vrf = np.load(
            osp.join(config['landmark_path'], 'point3D_vrf_n{:d}_{:s}_{:s}.npy'.format(config['n_cluster'],
                                                                                       config['cluster_mode'],
                                                                                       config['cluster_method'])),
            allow_pickle=True)[()]

        # construct 3D map
        self.initialize_point3Ds(p3ds=p3ds, p3d_descs=p3d_descs, p3d_seg=p3d_seg)
        self.initialize_ref_frames(cameras=cameras, images=images)

        all_vrf_frame_ids = []
        self.seg_ref_frame_ids = {}
        for sid in seg_vrf.keys():
            self.seg_ref_frame_ids[sid] = []
            for vi in seg_vrf[sid].keys():
                vrf_frame_id = seg_vrf[sid][vi]['image_id']
                if vrf_frame_id not in self.reference_frames.keys():
                    print(f'{vrf_frame_id} not in reference_frames')
                    continue
                self.seg_ref_frame_ids[sid].append(vrf_frame_id)
                if with_compress and vrf_frame_id in self.reference_frames.keys():
                    self.reference_frames[vrf_frame_id].point3D_ids = seg_vrf[sid][vi]['original_points3d']

            all_vrf_frame_ids.extend(self.seg_ref_frame_ids[sid])

        if with_compress:
            all_ref_ids = list(self.reference_frames.keys())
            for fid in all_ref_ids:
                valid = self.reference_frames[fid].associate_keypoints_with_point3Ds(point3Ds=self.point3Ds)
                if not valid:
                    del self.reference_frames[fid]

        all_vrf_frame_ids = np.unique(all_vrf_frame_ids)
        all_vrf_frame_ids = [v for v in all_vrf_frame_ids if v in self.reference_frames.keys()]
        self.build_covisibility_graph(frame_ids=all_vrf_frame_ids, n_frame=config['localization'][
            'covisibility_frame'])  # build covisible frames for vrf frames only

        logging.info(
            f'Construct {len(self.reference_frames.keys())} ref frames and {len(self.point3Ds.keys())} 3d points')

        self.gt_poses = {}
        if config['gt_pose_path'] is not None:
            gt_pose_path = osp.join(config['dataset_path'], config['gt_pose_path'])
            self.read_gt_pose(path=gt_pose_path)

    def read_gt_pose(self, path, prefix=''):
        self.gt_poses = read_gt_pose(path=path)
        print('Load {} gt poses'.format(len(self.gt_poses.keys())))

    def initialize_point3Ds(self, p3ds, p3d_descs, p3d_seg):
        self.point3Ds = {}
        for id in p3ds.keys():
            if id not in p3d_seg.keys():
                continue
            self.point3Ds[id] = Point3D(id=id, xyz=p3ds[id].xyz, error=p3ds[id].error,
                                        refframe_id=-1, rgb=p3ds[id].rgb,
                                        descriptor=p3d_descs[id], seg_id=p3d_seg[id],
                                        frame_ids=p3ds[id].image_ids)

    def initialize_ref_frames(self, cameras, images):
        self.reference_frames = {}
        for id in images.keys():
            im = images[id]
            cam = cameras[im.camera_id]
            self.reference_frames[id] = RefFrame(camera=cam,
                                                 id=id,
                                                 qvec=im.qvec,
                                                 tvec=im.tvec,
                                                 point3D_ids=im.point3D_ids,
                                                 keypoints=im.xys,
                                                 name=im.name)

    def localize_with_ref_frame(self, q_frame: Frame, q_kpt_ids: np.ndarray, sid, semantic_matching=False):
        ref_frame_id = self.seg_ref_frame_ids[sid][0]
        ref_frame = self.reference_frames[ref_frame_id]
        if semantic_matching and sid > 0:
            ref_data = ref_frame.get_keypoints_by_sid(sid=sid)
        else:
            ref_data = ref_frame.get_keypoints()

        q_descs = q_frame.descriptors[q_kpt_ids]
        q_kpts = q_frame.keypoints[q_kpt_ids, :2]
        q_scores = q_frame.keypoints[q_kpt_ids, 2]

        xyzs = ref_data['xyzs']
        point3D_ids = ref_data['point3D_ids']
        ref_sids = np.array([self.point3Ds[v].seg_id for v in point3D_ids])
        with torch.no_grad():
            indices0 = self.matcher({
                'descriptors0': torch.from_numpy(q_descs)[None].cuda().float(),
                'keypoints0': torch.from_numpy(q_kpts)[None].cuda().float(),
                'scores0': torch.from_numpy(q_scores)[None].cuda().float(),
                'image_shape0': (1, 3, q_frame.camera.width, q_frame.camera.height),

                'descriptors1': torch.from_numpy(ref_data['descriptors'])[None].cuda().float(),
                'keypoints1': torch.from_numpy(ref_data['keypoints'])[None].cuda().float(),
                'scores1': torch.from_numpy(ref_data['scores'])[None].cuda().float(),
                'image_shape1': (1, 3, ref_frame.camera.width, ref_frame.camera.height),
            }
            )['matches0'][0].cpu().numpy()

        valid = indices0 >= 0
        mkpts = q_kpts[valid]
        mkpt_ids = q_kpt_ids[valid]
        mxyzs = xyzs[indices0[valid]]
        mpoint3D_ids = point3D_ids[indices0[valid]]
        matched_sids = ref_sids[indices0[valid]]
        matched_ref_keypoints = ref_data['keypoints'][indices0[valid]]

        # print('mkpts: ', mkpts.shape, mxyzs.shape, np.sum(indices0 >= 0))
        # cfg = q_frame.camera._asdict()
        # q_cam = pycolmap.Camera(model=q_frame.camera.model, )
        # config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
        ret = pycolmap.absolute_pose_estimation(mkpts + 0.5,
                                                mxyzs,
                                                q_frame.camera,
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
        ret['matched_keypoints'] = mkpts
        ret['matched_keypoint_ids'] = mkpt_ids
        ret['matched_xyzs'] = mxyzs
        ret['reference_frame_id'] = ref_frame_id
        ret['matched_point3D_ids'] = mpoint3D_ids
        ret['matched_sids'] = matched_sids
        ret['matched_ref_keypoints'] = matched_ref_keypoints

        if not ret['success']:
            ret['num_inliers'] = 0
            ret['inliers'] = np.zeros(shape=(mkpts.shape[0],), dtype=bool)
        return ret

    def match(self, query_data, ref_data):
        q_descs = query_data['descriptors']
        q_kpts = query_data['keypoints']
        q_scores = query_data['scores']
        xyzs = ref_data['xyzs']
        points3D_ids = ref_data['point3D_ids']
        with torch.no_grad():
            indices0 = self.matcher({
                'descriptors0': torch.from_numpy(q_descs)[None].cuda().float(),
                'keypoints0': torch.from_numpy(q_kpts)[None].cuda().float(),
                'scores0': torch.from_numpy(q_scores)[None].cuda().float(),
                'image_shape0': (1, 3, query_data['camera'].width, query_data['camera'].height),

                'descriptors1': torch.from_numpy(ref_data['descriptors'])[None].cuda().float(),
                'keypoints1': torch.from_numpy(ref_data['keypoints'])[None].cuda().float(),
                'scores1': torch.from_numpy(ref_data['scores'])[None].cuda().float(),
                'image_shape1': (1, 3, ref_data['camera'].width, ref_data['camera'].height),
            }
            )['matches0'][0].cpu().numpy()

        valid = indices0 >= 0
        mkpts = q_kpts[valid]
        mkpt_ids = np.where(valid)[0]
        mxyzs = xyzs[indices0[valid]]
        mpoints3D_ids = points3D_ids[indices0[valid]]

        return {
            'matched_keypoints': mkpts,
            'matched_xyzs': mxyzs,
            'matched_point3D_ids': mpoints3D_ids,
            'matched_keypoint_ids': mkpt_ids,
        }

    def build_covisibility_graph(self, frame_ids: list = None, n_frame: int = 20):
        def find_covisible_frames(frame_id):
            observed = self.reference_frames[frame_id].point3D_ids
            covis = defaultdict(int)
            for pid in observed:
                if pid == -1:
                    continue
                if pid not in self.point3Ds.keys():
                    continue
                for img_id in self.point3Ds[pid].frame_ids:
                    covis[img_id] += 1

            covis_ids = np.array(list(covis.keys()))
            covis_num = np.array([covis[i] for i in covis_ids])

            if len(covis_ids) <= n_frame:
                sel_covis_ids = covis_ids[np.argsort(-covis_num)]
            else:
                ind_top = np.argpartition(covis_num, -n_frame)
                ind_top = ind_top[-n_frame:]  # unsorted top k
                ind_top = ind_top[np.argsort(-covis_num[ind_top])]
                sel_covis_ids = [covis_ids[i] for i in ind_top]

            return sel_covis_ids

        if frame_ids is None:
            frame_ids = list(self.referece_frames.keys())

        self.covisible_graph = defaultdict()
        for frame_id in frame_ids:
            self.covisible_graph[frame_id] = find_covisible_frames(frame_id=frame_id)

    def refine_pose(self, q_frame: Frame, refinement_method='matching'):
        if refinement_method == 'matching':
            return self.refine_pose_by_matching(q_frame=q_frame)
        elif refinement_method == 'projection':
            return self.refine_pose_by_projection(q_frame=q_frame)
        else:
            raise NotImplementedError

    def refine_pose_by_matching(self, q_frame):
        ref_frame_id = q_frame.reference_frame_id
        db_ids = self.covisible_graph[ref_frame_id]
        print('Find {} covisible frames'.format(len(db_ids)))
        loc_success = q_frame.tracking_status
        if loc_success and ref_frame_id in db_ids:
            init_kpts = q_frame.matched_keypoints
            init_kpt_ids = q_frame.matched_keypoint_ids
            init_point3D_ids = q_frame.matched_point3D_ids
            init_xyzs = np.array([self.point3Ds[v].xyz for v in init_point3D_ids]).reshape(-1, 3)
            list(db_ids).remove(ref_frame_id)
        else:
            init_kpts = None
            init_xyzs = None
            init_point3D_ids = None

        matched_xyzs = []
        matched_kpts = []
        matched_point3D_ids = []
        matched_kpt_ids = []
        for idx, frame_id in enumerate(db_ids):
            ref_data = self.reference_frames[frame_id].get_keypoints()
            match_out = self.match(query_data={
                'keypoints': q_frame.keypoints[:, :2],
                'scores': q_frame.keypoints[:, 2],
                'descriptors': q_frame.descriptors,
                'camera': q_frame.camera, },
                ref_data=ref_data)
            if match_out['matched_keypoints'].shape[0] > 0:
                matched_kpts.append(match_out['matched_keypoints'])
                matched_xyzs.append(match_out['matched_xyzs'])
                matched_point3D_ids.append(match_out['matched_point3D_ids'])
                matched_kpt_ids.append(match_out['matched_keypoint_ids'])
        if len(matched_kpts) > 1:
            matched_kpts = np.vstack(matched_kpts)
            matched_xyzs = np.vstack(matched_xyzs).reshape(-1, 3)
            matched_point3D_ids = np.hstack(matched_point3D_ids)
            matched_kpt_ids = np.hstack(matched_kpt_ids)
        else:
            matched_kpts = matched_kpts[0]
            matched_xyzs = matched_xyzs[0]
            matched_point3D_ids = matched_point3D_ids[0]
            matched_kpt_ids = matched_kpt_ids[0]
        if init_kpts is not None and init_kpts.shape[0] > 0:
            matched_kpts = np.vstack([matched_kpts, init_kpts])
            matched_xyzs = np.vstack([matched_xyzs, init_xyzs])
            matched_point3D_ids = np.hstack([matched_point3D_ids, init_point3D_ids])
            matched_kpt_ids = np.hstack([matched_kpt_ids, init_kpt_ids])

        matched_sids = np.array([self.point3Ds[v].seg_id for v in matched_point3D_ids])

        print_text = 'Refinement by matching. Get {:d} covisible frames with {:d} matches for optimization'.format(
            len(db_ids), matched_xyzs.shape[0])
        print(print_text)

        t_start = time.time()
        ret = pycolmap.absolute_pose_estimation(matched_kpts + 0.5,
                                                matched_xyzs,
                                                q_frame.camera,
                                                estimation_options={
                                                    'ransac': {
                                                        'max_error': self.config['localization']['threshold'],
                                                        'min_num_trials': 1000,
                                                        'max_num_trials': 10000,
                                                        'confidence': 0.995,
                                                    }},
                                                refinement_options={},
                                                # max_error_px=self.config['localization']['threshold'],
                                                # min_num_trials=1000, max_num_trials=10000, confidence=0.995)
                                                )
        print('Time of RANSAC: {:.2f}s'.format(time.time() - t_start))

        if ret is None:
            ret = {'success': False, }
        else:
            ret['success'] = True
            ret['qvec'] = ret['cam_from_world'].rotation.quat[[3, 0, 1, 2]]
            ret['tvec'] = ret['cam_from_world'].translation

        ret['matched_keypoints'] = matched_kpts
        ret['matched_keypoint_ids'] = matched_kpt_ids
        ret['matched_xyzs'] = matched_xyzs
        ret['matched_point3D_ids'] = matched_point3D_ids
        ret['matched_sids'] = matched_sids

        if ret['success']:
            inlier_mask = np.array(ret['inliers'])
            best_reference_frame_ids = self.find_reference_frames(matched_point3D_ids=matched_point3D_ids[inlier_mask],
                                                                  candidate_frame_ids=self.covisible_graph.keys())
        else:
            best_reference_frame_ids = self.find_reference_frames(matched_point3D_ids=matched_point3D_ids,
                                                                  candidate_frame_ids=self.covisible_graph.keys())

        ret['refinement_reference_frame_ids'] = best_reference_frame_ids[:self.config['localization'][
            'covisibility_frame']]
        ret['reference_frame_id'] = best_reference_frame_ids[0]

        return ret

    @torch.no_grad()
    def refine_pose_by_projection(self, q_frame):
        q_Rcw = qvec2rotmat(q_frame.qvec)
        q_tcw = q_frame.tvec
        q_Tcw = np.eye(4, dtype=float)  # [4 4]
        q_Tcw[:3, :3] = q_Rcw
        q_Tcw[:3, 3] = q_tcw
        cam = q_frame.camera
        imw = cam.width
        imh = cam.height
        K = q_frame.get_intrinsics()  # [3, 3]
        reference_frame_id = q_frame.reference_frame_id
        covis_frame_ids = self.covisible_graph[reference_frame_id]
        if reference_frame_id not in covis_frame_ids:
            covis_frame_ids.append(reference_frame_id)
        all_point3D_ids = []

        for frame_id in covis_frame_ids:
            all_point3D_ids.extend(list(self.reference_frames[frame_id].point3D_ids))

        all_point3D_ids = np.unique(all_point3D_ids)
        all_xyzs = []
        all_descs = []
        all_sids = []
        for pid in all_point3D_ids:
            all_xyzs.append(self.point3Ds[pid].xyz)
            all_descs.append(self.point3Ds[pid].descriptor)
            all_sids.append(self.point3Ds[pid].seg_id)

        all_xyzs = np.array(all_xyzs)  # [N 3]
        all_descs = np.array(all_descs)  # [N 3]
        all_point3D_ids = np.array(all_point3D_ids)
        all_sids = np.array(all_sids)

        # move to gpu (distortion is not included)
        # proj_uv = pycolmap.camera.img_from_cam(
        #     np.array([1, 1, 1]).reshape(1, 3),
        # )
        all_xyzs_cuda = torch.from_numpy(all_xyzs).cuda()
        ones = torch.ones(size=(all_xyzs_cuda.shape[0], 1), dtype=all_xyzs_cuda.dtype).cuda()
        all_xyzs_cuda_homo = torch.cat([all_xyzs_cuda, ones], dim=1)  # [N 4]
        K_cuda = torch.from_numpy(K).cuda()
        proj_uvs = K_cuda @ (torch.from_numpy(q_Tcw).cuda() @ all_xyzs_cuda_homo.t())[:3, :]  # [3, N]
        proj_uvs[0] /= proj_uvs[2]
        proj_uvs[1] /= proj_uvs[2]
        mask = (proj_uvs[2] > 0) * (proj_uvs[2] < 100) * (proj_uvs[0] >= 0) * (proj_uvs[0] < imw) * (
                proj_uvs[1] >= 0) * (proj_uvs[1] < imh)

        proj_uvs = proj_uvs[:, mask]

        print('Projection: out of range {:d}/{:d}'.format(all_xyzs_cuda.shape[0], proj_uvs.shape[1]))

        mxyzs = all_xyzs[mask.cpu().numpy()]
        mpoint3D_ids = all_point3D_ids[mask.cpu().numpy()]
        msids = all_sids[mask.cpu().numpy()]

        q_kpts_cuda = torch.from_numpy(q_frame.keypoints[:, :2]).cuda()
        proj_error = q_kpts_cuda[..., None] - proj_uvs[:2][None]
        proj_error = torch.sqrt(torch.sum(proj_error ** 2, dim=1))  # [M N]
        out_of_range_mask = (proj_error >= 2 * self.config['localization']['threshold'])

        q_descs_cuda = torch.from_numpy(q_frame.descriptors).cuda().float()  # [M D]
        all_descs_cuda = torch.from_numpy(all_descs).cuda().float()[mask]  # [N D]
        desc_dist = torch.sqrt(2 - 2 * q_descs_cuda @ all_descs_cuda.t() + 1e-6)
        desc_dist[out_of_range_mask] = desc_dist[out_of_range_mask] + 100
        dists, ids = torch.topk(desc_dist, k=2, largest=False, dim=1)
        # apply nn ratio
        ratios = dists[:, 0] / dists[:, 1]  # smaller, better
        ratio_mask = (ratios <= 0.995) * (dists[:, 0] < 100)
        ratio_mask = ratio_mask.cpu().numpy()
        ids = ids.cpu().numpy()[ratio_mask, 0]

        ratio_num = torch.sum(ratios <= 0.995)
        proj_num = torch.sum(dists[:, 0] < 100)

        print('Projection: after ratio {:d}/{:d}, ratio {:d}, proj {:d}'.format(q_kpts_cuda.shape[0],
                                                                                np.sum(ratio_mask),
                                                                                ratio_num, proj_num))

        mkpts = q_frame.keypoints[ratio_mask]
        mkpt_ids = np.where(ratio_mask)[0]
        mxyzs = mxyzs[ids]
        mpoint3D_ids = mpoint3D_ids[ids]
        msids = msids[ids]
        print('projection: ', mkpts.shape, mkpt_ids.shape, mxyzs.shape, mpoint3D_ids.shape, msids.shape)

        t_start = time.time()
        ret = pycolmap.absolute_pose_estimation(mkpts[:, :2] + 0.5, mxyzs, q_frame.camera,
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
        # inlier_mask = np.ones(shape=(mkpts.shape[0],), dtype=bool).tolist()
        # ret = pycolmap.pose_refinement(q_frame.tvec, q_frame.qvec, mkpts[:, :2] + 0.5, mxyzs, inlier_mask, cfg)
        # ret['num_inliers'] = np.sum(inlier_mask).astype(int)
        # ret['inliers'] = np.array(inlier_mask)

        print_text = 'Refinement by projection. Get {:d} inliers of {:d} matches for optimization'.format(
            ret['num_inliers'], mxyzs.shape[0])
        print(print_text)
        print('Time of RANSAC: {:.2f}s'.format(time.time() - t_start))

        ret['matched_keypoints'] = mkpts
        ret['matched_xyzs'] = mxyzs
        ret['matched_point3D_ids'] = mpoint3D_ids
        ret['matched_sids'] = msids
        ret['matched_keypoint_ids'] = mkpt_ids

        if ret['success']:
            inlier_mask = np.array(ret['inliers'])
            best_reference_frame_ids = self.find_reference_frames(matched_point3D_ids=mpoint3D_ids[inlier_mask],
                                                                  candidate_frame_ids=self.covisible_graph.keys())
        else:
            best_reference_frame_ids = self.find_reference_frames(matched_point3D_ids=mpoint3D_ids,
                                                                  candidate_frame_ids=self.covisible_graph.keys())

        ret['refinement_reference_frame_ids'] = best_reference_frame_ids[:self.config['localization'][
            'covisibility_frame']]
        ret['reference_frame_id'] = best_reference_frame_ids[0]

        if not ret['success']:
            ret['num_inliers'] = 0
            ret['inliers'] = np.zeros(shape=(mkpts.shape[0],), dtype=bool)

        return ret

    def find_reference_frames(self, matched_point3D_ids, candidate_frame_ids=None):
        covis_frames = defaultdict(int)
        for pid in matched_point3D_ids:
            for im_id in self.point3Ds[pid].frame_ids:
                if candidate_frame_ids is not None and im_id in candidate_frame_ids:
                    covis_frames[im_id] += 1

        covis_ids = np.array(list(covis_frames.keys()))
        covis_num = np.array([covis_frames[i] for i in covis_ids])
        sorted_idxes = np.argsort(covis_num)[::-1]  # larger to small
        sorted_frame_ids = covis_ids[sorted_idxes]
        return sorted_frame_ids

    def check_semantic_consistency(self, q_frame: Frame, sid, overlap_ratio=0.5):
        ref_frame_id = self.seg_ref_frame_ids[sid][0]
        ref_frame = self.reference_frames[ref_frame_id]

        q_sids = q_frame.seg_ids
        ref_sids = np.array([self.point3Ds[v].seg_id for v in ref_frame.point3D_ids]) + self.start_sid
        overlap_sids = np.intersect1d(q_sids, ref_sids)

        overlap_num1 = 0
        overlap_num2 = 0
        for sid in overlap_sids:
            overlap_num1 += np.sum(q_sids == sid)
            overlap_num2 += np.sum(ref_sids == sid)

        ratio1 = overlap_num1 / q_sids.shape[0]
        ratio2 = overlap_num2 / ref_sids.shape[0]

        # print('semantic_check: ', overlap_sids, overlap_num1, ratio1, overlap_num2, ratio2)

        return min(ratio1, ratio2) >= overlap_ratio
