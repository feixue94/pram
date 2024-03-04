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
from localization.refframe import RefFrame
from localization.point3d import Point3D
from colmap_utils.read_write_model import qvec2rotmat, read_model, read_compressed_model


class SingleMap3D:
    def __init__(self, config, with_compress=False):
        self.config = config
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
        self.seg_vrf = seg_vrf  # [seg_id, ref_id]
        vrf_ids = []
        for sid in self.seg_vrf.keys():
            vrf_ids.extend([seg_vrf[sid][v]['image_id'] for v in seg_vrf[sid].keys()])

        vrf_ids = np.unique(vrf_ids)
        vrf_ids = [v for v in vrf_ids if v in self.ref_frames.keys()]
        self.build_covisibility_graph(frame_ids=vrf_ids, n_frame=config['localization'][
            'covisibility_frame'])  # build covisible frames for vrf frames only

        logging.info(f'Construct {len(self.ref_frames.keys())} ref frames and {len(self.point3ds.keys())} 3d points')

    def initialize_point3Ds(self, p3ds, p3d_descs, p3d_seg):
        self.point3ds = {}
        for id in p3ds.keys():
            if id not in p3d_seg.keys():
                continue
            self.point3ds[id] = Point3D(id=id, xyz=p3ds[id].xyz, error=p3ds[id].error, refframe_id=-1,
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

    def build_covisibility_graph(self, frame_ids: list = None, n_frame: int = 20):
        def find_covisible_frames(frame_id):
            observed = self.ref_frames[frame_id].point3D_ids
            covis = defaultdict(int)
            for pid in observed:
                if pid == -1:
                    continue
                if pid not in self.point3ds.keys():
                    continue
                for img_id in self.point3ds[pid].frame_ids:
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
