# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> map3d
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/03/2024 10:25
=================================================='''
import numpy as np
from collections import defaultdict
import os
import os.path as osp
import pycolmap
import logging
import time

import torch

from localization.refframe import RefFrame
from localization.point3d import Point3D
from colmap_utils.read_write_model import qvec2rotmat, read_model, read_compressed_model
from localization.utils import read_gt_pose


class SingleMap3D:
    def __init__(self, config, matcher, with_compress=False):
        self.config = config
        self.matcer = matcher
        self.image_path_prefix = self.config['image_path_prefix']
        if not with_compress:
            cameras, images, p3ds = read_model(
                path=osp.join(config['segment_path'], 'model'), ext='.bin'
            )
            p3d_descs = np.load(osp.join(config['segment_path'], 'point3D_desc.npy'),
                                allow_pickle=True)[()]
        else:
            cameras, images, p3ds = read_compressed_model(
                path=osp.join(config['segment_path'], 'compress_model_{:s}'.format(config['cluster_method'])),
                ext='.bin')
            p3d_descs = np.load(osp.join(config['segment_path'], 'compress_model_{:s}/point3D_desc.npy'.format(
                config['cluster_method'])), allow_pickle=True)[()]

        print('Load {} cameras {} images {} 3D points'.format(len(cameras), len(images), len(p3d_descs)))

        seg_data = np.load(
            osp.join(config['segment_path'], 'point3D_cluster_n{:d}_{:s}_{:s}.npy'.format(config['n_cluster'],
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
            osp.join(config['segment_path'], 'point3D_vrf_n{:d}_{:s}_{:s}.npy'.format(config['n_cluster'],
                                                                                      config['cluster_mode'],
                                                                                      config['cluster_method'])),
            allow_pickle=True)[()]

        # construct 3D map
        self.initialize_point3Ds(p3ds=p3ds, p3d_descs=p3d_descs, p3d_seg=p3d_seg)
        self.initialize_ref_frames(cameras=cameras, images=images)
        vrf_ids = []
        self.seg_ref_frame_ids = {}
        for sid in seg_vrf.keys():
            self.seg_ref_frame_ids[sid] = [seg_vrf[sid][v]['image_id'] for v in seg_vrf[sid].keys()]
            vrf_ids.extend([seg_vrf[sid][v]['image_id'] for v in seg_vrf[sid].keys()])

        vrf_ids = np.unique(vrf_ids)
        vrf_ids = [v for v in vrf_ids if v in self.ref_frames.keys()]
        self.build_covisibility_graph(frame_ids=vrf_ids, n_frame=config['localization'][
            'covisibility_frame'])  # build covisible frames for vrf frames only

        logging.info(f'Construct {len(self.ref_frames.keys())} ref frames and {len(self.point3Ds.keys())} 3d points')

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
            self.point3Ds[id] = Point3D(id=id, xyz=p3ds[id].xyz, error=p3ds[id].error, refframe_id=-1,
                                        descriptor=p3d_descs[id], seg_id=p3d_seg[id], frame_ids=p3ds[id].image_ids)

    def initialize_ref_frames(self, cameras, images):
        self.ref_frames = {}
        for id in images.keys():
            im = images[id]
            cam = cameras[im.camera_id]
            self.ref_frames[id] = RefFrame(camera=cam, id=id, qvec=im.qvec, tvec=im.tvec,
                                           point3D_ids=im.point3D_ids,
                                           keypoints=im.xys,
                                           name=im.name)

    def localize_with_ref_frame(self, query_data, sid, semantic_matching=False):
        ref_frame_id = self.seg_ref_frame_ids[sid][0]
        ref_frame = self.ref_frames[ref_frame_id]
        if semantic_matching and sid > 0:
            ref_data = ref_frame.get_keypoints_by_sid(sid=sid, point3Ds=self.point3Ds)
        else:
            ref_data = ref_frame.get_keypoints(point3Ds=self.point3Ds)

        q_descs = query_data['descriptors']
        q_kpts = query_data['keypoints']
        q_scores = query_data['scores']
        xyzs = ref_data['xyzs']
        with torch.no_grad():
            indices0 = self.matcer({
                'descriptors0': torch.from_numpy(q_descs)[None].cuda().float(),
                'keypoints0': torch.from_numpy(q_kpts)[None].cuda().float(),
                'scores0': torch.from_numpy(q_scores)[None].cuda().float(),
                'image_shape0': (1, 3, query_data['camera'].width, query_data['camera'].height),

                'descriptors1': torch.from_numpy(ref_data['descriptors'])[None].cuda().float(),
                'keypoints1': torch.from_numpy(ref_data['keypoints'])[None].cuda().float(),
                'scores1': torch.from_numpy(ref_data['scores'])[None].cuda().float(),
                'image_shape1': (1, 3, ref_frame.camera.width, ref_frame.camera.height),
            }
            )['matches0'][0].cpu().numpy()

        mkqs = q_kpts[indices0 >= 0]
        mxyzs = xyzs[indices0[indices0 >= 0]]

        print('mkqs: ', mkqs.shape, mxyzs.shape, np.sum(indices0 >= 0))

        cfg = {
            'model': query_data['camera'].model,
            'width': query_data['camera'].width,
            'height': query_data['camera'].height,
            'params': query_data['camera'].params,
        }
        ret = pycolmap.absolute_pose_estimation(mkqs + 0.5, mxyzs, cfg, 12)
        ret['matched_kpts'] = mkqs
        ret['matched_xyzs'] = mxyzs
        ret['ref_frame_id'] = ref_frame_id

        return ret

    def match(self, query_data, ref_data):
        q_descs = query_data['descriptors']
        q_kpts = query_data['keypoints']
        q_scores = query_data['scores']
        xyzs = ref_data['xyzs']
        with torch.no_grad():
            indices0 = self.matcer({
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

        mkqs = q_kpts[indices0 >= 0]
        mxyzs = xyzs[indices0[indices0 >= 0]]

        return {
            'matched_kpts': mkqs,
            'matched_xyzs': mxyzs,
        }

    def build_covisibility_graph(self, frame_ids: list = None, n_frame: int = 20):
        def find_covisible_frames(frame_id):
            observed = self.ref_frames[frame_id].point3D_ids
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
            frame_ids = list(self.ref_frames.keys())

        self.covisible_graph = defaultdict()
        for frame_id in frame_ids:
            self.covisible_graph[frame_id] = find_covisible_frames(frame_id=frame_id)

    def refine_pose(self, query_data, ref_frame_id, refinement_method='matching'):
        if refinement_method == 'matching':
            return self.refine_pose_by_matching(query_data=query_data, ref_frame_id=ref_frame_id)
        elif refinement_method == 'projection':
            return self.refine_pose_by_projection(query_data=query_data, ref_frame_id=ref_frame_id)
        else:
            raise NotImplementedError

    def refine_pose_by_matching(self, query_data, ref_frame_id):
        db_ids = self.covisible_graph[ref_frame_id]
        print('Find {} covisible frames'.format(len(db_ids)))
        loc_success = query_data['loc_success']
        if loc_success and ref_frame_id in db_ids:
            init_kpts = query_data['matched_kpts']
            init_xyzs = query_data['matched_xyzs']
            db_ids.remove(ref_frame_id)
        else:
            init_kpts = None
            init_xyzs = None

        matched_xyzs = []
        matched_kpts = []
        for idx, frame_id in enumerate(db_ids):
            ref_data = self.ref_frames[frame_id].get_keypoints(point3Ds=self.point3Ds)
            match_out = self.match(query_data=query_data, ref_data=ref_data)
            if match_out['matched_kpts'].shape[0] > 0:
                matched_kpts.append(match_out['matched_kpts'])
                matched_xyzs.append(match_out['matched_xyzs'])

        matched_kpts = np.vstack(matched_kpts)
        matched_xyzs = np.vstack(matched_xyzs).reshape(-1, 3)
        if init_kpts is not None:
            matched_kpts = np.vstack([matched_kpts, init_kpts])
            matched_xyzs = np.vstack([matched_xyzs, init_xyzs])

        print_text = 'Refinement by matching. Get {:d} covisible frames with {:d} matches for optimization'.format(
            len(db_ids), matched_xyzs.shape[0])
        print(print_text)

        t_start = time.time()
        cfg = {
            'model': query_data['camera'].model,
            'width': query_data['camera'].width,
            'height': query_data['camera'].height,
            'params': query_data['camera'].params,
        }
        ret = pycolmap.absolute_pose_estimation(matched_kpts + 0.5, matched_xyzs, cfg,
                                                max_error_px=self.config['localization']['threshold'],
                                                min_num_trials=1000, max_num_trials=10000, confidence=0.995)
        print('Time of RANSAC: {:.2f}s'.format(time.time() - t_start))

        ret['matched_kpts'] = matched_kpts
        ret['matched_xyzs'] = matched_xyzs
        ret['reference_db_ids'] = db_ids

        return ret

    def refine_pose_by_projection(self, query_data, ref_frame_id):
        pass
