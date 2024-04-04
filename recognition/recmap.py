# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> recmap
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 11:02
=================================================='''

import torch
import os
import os.path as osp
import numpy as np
import cv2
import yaml
import multiprocessing as mp
from copy import deepcopy
import logging
import h5py
from tqdm import tqdm
import open3d as o3d
from sklearn.cluster import KMeans, Birch
from collections import defaultdict
from colmap_utils.read_write_model import read_model, qvec2rotmat, write_cameras_binary, write_images_binary
from colmap_utils.read_write_model import write_points3d_binary, Image, Point3D, Camera
from colmap_utils.read_write_model import write_compressed_points3d_binary, write_compressed_images_binary
from recognition.vis_seg import generate_color_dic, vis_seg_point, plot_kpts


class RecMap:
    def __init__(self):
        self.cameras = None
        self.images = None
        self.points3D = None
        self.pcd = o3d.geometry.PointCloud()
        self.seg_color_dict = generate_color_dic(n_seg=1000)

    def load_sfm_model(self, path: str, ext='.bin'):
        self.cameras, self.images, self.points3D = read_model(path, ext)
        self.name_to_id = {image.name: i for i, image in self.images.items()}
        print('Load {:d} cameras, {:d} images, {:d} points'.format(len(self.cameras), len(self.images),
                                                                   len(self.points3D)))

    def remove_statics_outlier(self, nb_neighbors: int = 20, std_ratio: float = 2.0):
        xyzs = []
        p3d_ids = []
        for p3d_id in self.points3D.keys():
            xyzs.append(self.points3D[p3d_id].xyz)
            p3d_ids.append(p3d_id)

        xyzs = np.array(xyzs)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        new_pcd, inlier_ids = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        new_point3Ds = {}
        for i in inlier_ids:
            new_point3Ds[p3d_ids[i]] = self.points3D[p3d_ids[i]]
        self.points3D = new_point3Ds
        n_outlier = xyzs.shape[0] - len(inlier_ids)
        ratio = n_outlier / xyzs.shape[0]
        print('Remove {:d} - {:d} = {:d}/{:.2f}% points'.format(xyzs.shape[0], len(inlier_ids), n_outlier, ratio * 100))

    def load_segmentation(self, path: str):
        data = np.load(path, allow_pickle=True)[()]
        p3d_id = data['id']
        seg_id = data['label']
        self.p3d_seg = {p3d_id[i]: seg_id[i] for i in range(p3d_id.shape[0])}
        self.seg_p3d = {}
        for pid in self.p3d_seg.keys():
            sid = self.p3d_seg[pid]
            if sid not in self.seg_p3d.keys():
                self.seg_p3d[sid] = [pid]
            else:
                self.seg_p3d[sid].append(pid)

        if 'xyz' not in data.keys():
            all_xyz = []
            for pid in p3d_id:
                xyz = self.points3D[pid].xyz
                all_xyz.append(xyz)
            data['xyz'] = np.array(all_xyz)
            np.save(path, data)
            print('Add xyz to ', path)

    def cluster(self, k=512, mode='xyz', min_obs=3, save_fn=None, method='kmeans', **kwargs):
        if save_fn is not None:
            if osp.isfile(save_fn):
                print('{:s} exists.'.format(save_fn))
                return
        all_xyz = []
        point3D_ids = []
        for p3d in self.points3D.values():
            track_len = len(p3d.point2D_idxs)
            if track_len < min_obs:
                continue
            all_xyz.append(p3d.xyz)
            point3D_ids.append(p3d.id)

        xyz = np.array(all_xyz)
        point3D_ids = np.array(point3D_ids)

        if mode.find('x') < 0:
            xyz[:, 0] = 0
        if mode.find('y') < 0:
            xyz[:, 1] = 0
        if mode.find('z') < 0:
            xyz[:, 2] = 0

        print('xyz: ', xyz[0:2])
        if method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=0, verbose=True).fit(xyz)
        elif method == 'birch':
            model = Birch(threshold=kwargs.get('threshold'), n_clusters=k).fit(xyz)  # 0.01 for indoor
        else:
            print('Method {:s} for clustering does not exist'.format(method))
            exit(0)
        labels = np.array(model.labels_).reshape(-1)
        if save_fn is not None:
            np.save(save_fn, {
                'id': np.array(point3D_ids),  # should be assigned to self.points3D_ids
                'label': np.array(labels),
                'xyz': np.array(all_xyz),
            })

    def assign_point3D_descriptor(self, feature_fn: str, save_fn=None, n_process=1):
        '''
        assign each 3d point a descriptor for localization
        :param feature_fn: file name of features [h5py]
        :param save_fn:
        :param n_process:
        :return:
        '''

        def run(start_id, end_id, points3D_desc):
            for pi in tqdm(range(start_id, end_id), total=end_id - start_id):
                p3d_id = all_p3d_ids[pi]
                img_list = self.points3D[p3d_id].image_ids
                kpt_ids = self.points3D[p3d_id].point2D_idxs
                all_descs = []
                for img_id, p2d_id in zip(img_list, kpt_ids):
                    if img_id not in self.images.keys():
                        continue
                    img_fn = self.images[img_id].name
                    desc = feat_file[img_fn]['descriptors'][()].transpose()[p2d_id]
                    all_descs.append(desc)

                if len(all_descs) == 1:
                    points3D_desc[p3d_id] = all_descs[0]
                else:
                    all_descs = np.array(all_descs)  # [n, d]
                    dist = all_descs @ all_descs.transpose()  # [n, n]
                    dist = 2 - 2 * dist
                    md_dist = np.median(dist, axis=-1)  # [n]
                    min_id = np.argmin(md_dist)
                    points3D_desc[p3d_id] = all_descs[min_id]

        if osp.isfile(save_fn):
            print('{:s} exists.'.format(save_fn))
            return
        p3D_desc = {}
        feat_file = h5py.File(feature_fn, 'r')
        all_p3d_ids = sorted(self.points3D.keys())

        if n_process > 1:
            if len(all_p3d_ids) <= n_process:
                run(start_id=0, end_id=len(all_p3d_ids), points3D_desc=p3D_desc)
            else:
                manager = mp.Manager()
                output = manager.dict()  # necessary otherwise empty
                n_sample_per_process = len(all_p3d_ids) // n_process
                jobs = []
                for i in range(n_process):
                    start_id = i * n_sample_per_process
                    if i == n_process - 1:
                        end_id = len(all_p3d_ids)
                    else:
                        end_id = (i + 1) * n_sample_per_process
                    p = mp.Process(
                        target=run,
                        args=(start_id, end_id, output),
                    )
                    jobs.append(p)
                    p.start()

                for p in jobs:
                    p.join()

                p3D_desc = {}
                for k in output.keys():
                    p3D_desc[k] = output[k]
        else:
            run(start_id=0, end_id=len(all_p3d_ids), points3D_desc=p3D_desc)

        if save_fn is not None:
            np.save(save_fn, p3D_desc)

    def reproject(self, img_id, xyzs):
        qvec = self.images[img_id].qvec
        Rcw = qvec2rotmat(qvec=qvec)
        tvec = self.images[img_id].tvec
        tcw = tvec.reshape(3, )
        Tcw = np.eye(4, dtype=float)
        Tcw[:3, :3] = Rcw
        Tcw[:3, 3] = tcw
        # intrinsics
        cam = self.cameras[self.images[img_id].camera_id]
        K = self.get_intrinsics_from_camera(camera=cam)

        xyzs_homo = np.hstack([xyzs, np.ones(shape=(xyzs.shape[0], 1), dtype=float)])
        kpts = K @ ((Tcw @ xyzs_homo.transpose())[:3, :])  # [3, N]
        kpts = kpts.transpose()  # [N, 3]
        kpts[:, 0] = kpts[:, 0] / kpts[:, 2]
        kpts[:, 1] = kpts[:, 1] / kpts[:, 2]

        return kpts

    def find_covisible_frame_ids(self, image_id, images, points3D):
        covis = defaultdict(int)
        p3d_ids = images[image_id].point3D_ids

        for pid in p3d_ids:
            if pid == -1:
                continue
            if pid not in points3D.keys():
                continue
            for im in points3D[pid].image_ids:
                covis[im] += 1

        covis_ids = np.array(list(covis.keys()))
        covis_num = np.array([covis[i] for i in covis_ids])
        ind_top = np.argsort(covis_num)[::-1]
        sorted_covis_ids = [covis_ids[i] for i in ind_top]
        return sorted_covis_ids

    def create_virtual_frame_3(self, save_fn=None, save_vrf_dir=None, show_time=-1, ignored_cameras=[],
                               min_cover_ratio=0.9,
                               depth_scale=1.2,
                               radius=15,
                               min_obs=120,
                               topk_imgs=500,
                               n_vrf=10,
                               covisible_frame=20,
                               **kwargs):
        def reproject(img_id, xyzs):
            qvec = self.images[img_id].qvec
            Rcw = qvec2rotmat(qvec=qvec)
            tvec = self.images[img_id].tvec
            tcw = tvec.reshape(3, )
            Tcw = np.eye(4, dtype=float)
            Tcw[:3, :3] = Rcw
            Tcw[:3, 3] = tcw
            # intrinsics
            cam = self.cameras[self.images[img_id].camera_id]
            K = self.get_intrinsics_from_camera(camera=cam)

            xyzs_homo = np.hstack([xyzs, np.ones(shape=(xyzs.shape[0], 1), dtype=float)])
            kpts = K @ ((Tcw @ xyzs_homo.transpose())[:3, :])  # [3, N]
            kpts = kpts.transpose()  # [N, 3]
            kpts[:, 0] = kpts[:, 0] / kpts[:, 2]
            kpts[:, 1] = kpts[:, 1] / kpts[:, 2]

            return kpts

        def find_best_vrf_by_covisibility(p3d_id_list):
            all_img_ids = []
            all_xyzs = []

            img_ids_full = []
            img_id_obs = {}
            for pid in p3d_id_list:
                if pid not in self.points3D.keys():
                    continue
                all_xyzs.append(self.points3D[pid].xyz)

                img_ids = self.points3D[pid].image_ids
                for iid in img_ids:
                    if iid in all_img_ids:
                        continue
                    # valid_p3ds = [v for v in self.images[iid].point3D_ids if v > 0 and v in p3d_id_list]
                    if len(ignored_cameras) > 0:
                        ignore = False
                        img_name = self.images[iid].name
                        for c in ignored_cameras:
                            if img_name.find(c) >= 0:
                                ignore = True
                                break
                        if ignore:
                            continue
                    # valid_p3ds = np.intersect1d(np.array(self.images[iid].point3D_ids), np.array(p3d_id_list)).tolist()
                    valid_p3ds = [v for v in self.images[iid].point3D_ids if v > 0]
                    img_ids_full.append(iid)
                    if len(valid_p3ds) < min_obs:
                        continue

                    all_img_ids.append(iid)
                    img_id_obs[iid] = len(valid_p3ds)
            all_xyzs = np.array(all_xyzs)

            print('Find {} 3D points and {} images'.format(len(p3d_id_list), len(img_id_obs.keys())))
            top_img_ids_by_obs = sorted(img_id_obs.items(), key=lambda item: item[1], reverse=True)  # [(key, value), ]
            all_img_ids = []
            for item in top_img_ids_by_obs:
                all_img_ids.append(item[0])
                if len(all_img_ids) >= topk_imgs:
                    break

            # all_img_ids = all_img_ids[:200]
            if len(all_img_ids) == 0:
                print('no valid img ids with obs over {:d}'.format(min_obs))
                all_img_ids = img_ids_full

            img_observations = {}
            p3d_id_array = np.array(p3d_id_list)
            for idx, img_id in enumerate(all_img_ids):
                valid_p3ds = [v for v in self.images[img_id].point3D_ids if v > 0]
                mask = np.array([False for i in range(len(p3d_id_list))])
                for pid in valid_p3ds:
                    found_idx = np.where(p3d_id_array == pid)[0]
                    if found_idx.shape[0] == 0:
                        continue
                    mask[found_idx[0]] = True

                img_observations[img_id] = mask

            unobserved_p3d_ids = np.array([True for i in range(len(p3d_id_list))])

            candidate_img_ids = []
            total_cover_ratio = 0
            while total_cover_ratio < min_cover_ratio:
                best_img_id = -1
                best_img_obs = -1
                for idx, im_id in enumerate(all_img_ids):
                    if im_id in candidate_img_ids:
                        continue
                    obs_i = np.sum(img_observations[im_id] * unobserved_p3d_ids)
                    if obs_i > best_img_obs:
                        best_img_id = im_id
                        best_img_obs = obs_i

                if best_img_id >= 0:
                    # keep the valid img_id
                    candidate_img_ids.append(best_img_id)
                    # update the unobserved mask
                    unobserved_p3d_ids[img_observations[best_img_id]] = False
                    total_cover_ratio = 1 - np.sum(unobserved_p3d_ids) / len(p3d_id_list)
                    print(len(candidate_img_ids), best_img_obs, best_img_obs / len(p3d_id_list), total_cover_ratio)

                    if best_img_obs / len(p3d_id_list) < 0.01:
                        break

                    if len(candidate_img_ids) >= n_vrf:
                        break
                else:
                    break

            return candidate_img_ids
            # return [(v, img_observations[v]) for v in candidate_img_ids]

        if save_vrf_dir is not None:
            os.makedirs(save_vrf_dir, exist_ok=True)

        seg_ref = {}
        for sid in self.seg_p3d.keys():
            if sid == -1:  # ignore invalid segment
                continue
            all_p3d_ids = self.seg_p3d[sid]
            candidate_img_ids = find_best_vrf_by_covisibility(p3d_id_list=all_p3d_ids)

            seg_ref[sid] = {}
            for can_idx, img_id in enumerate(candidate_img_ids):
                cam = self.cameras[self.images[img_id].camera_id]
                width = cam.width
                height = cam.height
                qvec = self.images[img_id].qvec
                tvec = self.images[img_id].tvec

                img_name = self.images[img_id].name
                orig_p3d_ids = [p for p in self.images[img_id].point3D_ids if p in self.points3D.keys() and p >= 0]
                orig_xyzs = []
                new_xyzs = []
                for pid in all_p3d_ids:
                    if pid in orig_p3d_ids:
                        orig_xyzs.append(self.points3D[pid].xyz)
                    else:
                        if pid in self.points3D.keys():
                            new_xyzs.append(self.points3D[pid].xyz)

                if len(orig_xyzs) == 0:
                    continue

                orig_xyzs = np.array(orig_xyzs)
                new_xyzs = np.array(new_xyzs)

                print('img: ', osp.join(kwargs.get('image_root'), img_name))
                img = cv2.imread(osp.join(kwargs.get('image_root'), img_name))
                orig_kpts = reproject(img_id=img_id, xyzs=orig_xyzs)
                max_depth = depth_scale * np.max(orig_kpts[:, 2])
                orig_kpts = orig_kpts[:, :2]
                mask_ori = (orig_kpts[:, 0] >= 0) & (orig_kpts[:, 0] < width) & (orig_kpts[:, 1] >= 0) & (
                        orig_kpts[:, 1] < height)
                orig_kpts = orig_kpts[mask_ori]

                if orig_kpts.shape[0] == 0:
                    continue

                img_kpt = plot_kpts(img=img, kpts=orig_kpts, radius=[3 for i in range(orig_kpts.shape[0])],
                                    colors=[(0, 0, 255) for i in range(orig_kpts.shape[0])], thickness=-1)
                if new_xyzs.shape[0] == 0:
                    img_all = img_kpt
                else:
                    new_kpts = reproject(img_id=img_id, xyzs=new_xyzs)
                    mask_depth = (new_kpts[:, 2] > 0) & (new_kpts[:, 2] <= max_depth)
                    mask_in_img = (new_kpts[:, 0] >= 0) & (new_kpts[:, 0] < width) & (new_kpts[:, 1] >= 0) & (
                            new_kpts[:, 1] < height)
                    dist_all_orig = torch.from_numpy(new_kpts[:, :2])[..., None] - \
                                    torch.from_numpy(orig_kpts[:, :2].transpose())[None]
                    dist_all_orig = torch.sqrt(torch.sum(dist_all_orig ** 2, dim=1))  # [N, M]
                    min_dist = torch.min(dist_all_orig, dim=1)[0].numpy()
                    mask_close_to_img = (min_dist <= radius)

                    mask_new = (mask_depth & mask_in_img & mask_close_to_img)

                    cover_ratio = np.sum(mask_ori) + np.sum(mask_new)
                    cover_ratio = cover_ratio / len(all_p3d_ids)

                    print('idx: {:d}, img: ori {:d}/{:d}/{:.2f}, new {:d}/{:d}'.format(can_idx,
                                                                                       orig_kpts.shape[0],
                                                                                       np.sum(mask_ori),
                                                                                       cover_ratio * 100,
                                                                                       new_kpts.shape[0],
                                                                                       np.sum(mask_new)))

                    new_kpts = new_kpts[mask_new]

                    # img_all = img_kpt
                    img_all = plot_kpts(img=img_kpt, kpts=new_kpts, radius=[3 for i in range(new_kpts.shape[0])],
                                        colors=[(0, 255, 0) for i in range(new_kpts.shape[0])], thickness=-1)

                cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                cv2.imshow('img', img_all)

                if save_vrf_dir is not None:
                    cv2.imwrite(osp.join(save_vrf_dir,
                                         'seg-{:05d}_can-{:05d}_'.format(sid, can_idx) + img_name.replace('/', '+')),
                                img_all)

                key = cv2.waitKey(show_time)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(0)

                covisile_frame_ids = self.find_covisible_frame_ids(image_id=img_id, images=self.images,
                                                                   points3D=self.points3D)
                seg_ref[sid][can_idx] = {
                    'image_name': img_name,
                    'image_id': img_id,
                    'qvec': deepcopy(qvec),
                    'tvec': deepcopy(tvec),
                    'camera': {
                        'model': cam.model,
                        'params': cam.params,
                        'width': cam.width,
                        'height': cam.height,
                    },
                    'original_points3d': np.array(
                        [v for v in self.images[img_id].point3D_ids if v >= 0 and v in self.points3D.keys()]),
                    'covisible_frame_ids': np.array(covisile_frame_ids[:covisible_frame]),
                }
        # save vrf info
        if save_fn is not None:
            print('Save {} segments with virtual reference image information to {}'.format(len(seg_ref.keys()),
                                                                                           save_fn))
            np.save(save_fn, seg_ref)

    def visualize_3Dpoints(self):
        xyz = []
        rgb = []
        for point3D in self.points3D.values():
            xyz.append(point3D.xyz)
            rgb.append(point3D.rgb / 255)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

    def visualize_segmentation(self, p3d_segs, points3D):
        p3d_ids = p3d_segs.keys()
        xyzs = []
        rgbs = []
        for pid in p3d_ids:
            xyzs.append(points3D[pid].xyz)
            seg_color = self.seg_color_dict[p3d_segs[pid]]
            rgbs.append(np.array([seg_color[2], seg_color[1], seg_color[0]]) / 255)
        xyzs = np.array(xyzs)
        rgbs = np.array(rgbs)

        self.pcd.points = o3d.utility.Vector3dVector(xyzs)
        self.pcd.colors = o3d.utility.Vector3dVector(rgbs)

        o3d.visualization.draw_geometries([self.pcd])

    def visualize_segmentation_on_image(self, p3d_segs, image_path, feat_path):
        vis_color = generate_color_dic(n_seg=1024)
        feat_file = h5py.File(feat_path, 'r')

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        for mi in sorted(self.images.keys()):
            im = self.images[mi]
            im_name = im.name
            p3d_ids = im.point3D_ids
            p2ds = feat_file[im_name]['keypoints'][()]
            image = cv2.imread(osp.join(image_path, im_name))
            print('img_name: ', im_name)

            sems = []
            for pid in p3d_ids:
                if pid in p3d_segs.keys():
                    sems.append(p3d_segs[pid] + 1)
                else:
                    sems.append(0)
            sems = np.array(sems)

            sems = np.array(sems)
            mask = sems > 0
            img_seg = vis_seg_point(img=image, kpts=p2ds[mask], segs=sems[mask], seg_color=vis_color)

            cv2.imshow('img', img_seg)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit(0)
            elif key == ord('r'):
                # cv2.destroyAllWindows()
                return

    def extract_query_p3ds(self, log_fn, feat_fn, save_fn=None):
        if save_fn is not None:
            if osp.isfile(save_fn):
                print('{:s} exists'.format(save_fn))
                return

        loc_log = np.load(log_fn, allow_pickle=True)[()]
        fns = loc_log.keys()
        feat_file = h5py.File(feat_fn, 'r')

        out = {}
        for fn in tqdm(fns, total=len(fns)):
            matched_kpts = loc_log[fn]['keypoints_query']
            matched_p3ds = loc_log[fn]['points3D_ids']

            query_kpts = feat_file[fn]['keypoints'][()].astype(float)
            query_p3d_ids = np.zeros(shape=(query_kpts.shape[0],), dtype=int) - 1
            print('matched kpts: {}, query kpts: {}'.format(matched_kpts.shape[0], query_kpts.shape[0]))

            if matched_kpts.shape[0] > 0:
                # [M, 2, 1] - [1, 2, N] = [M, 2, N]
                dist = torch.from_numpy(matched_kpts).unsqueeze(-1) - torch.from_numpy(
                    query_kpts.transpose()).unsqueeze(0)
                dist = torch.sum(dist ** 2, dim=1)  # [M, N]
                values, idxes = torch.topk(dist, dim=1, largest=False, k=1)  # find the matches kpts with dist of 0
                values = values.numpy()
                idxes = idxes.numpy()
                for i in range(values.shape[0]):
                    if values[i, 0] < 1:
                        query_p3d_ids[idxes[i, 0]] = matched_p3ds[i]

            out[fn] = query_p3d_ids
        np.save(save_fn, out)
        feat_file.close()

    def compute_mean_scale_p3ds(self, min_obs=5, save_fn=None):
        if save_fn is not None:
            if osp.isfile(save_fn):
                with open(save_fn, 'r') as f:
                    lines = f.readlines()
                    l = lines[0].strip().split()
                    self.mean_xyz = np.array([float(v) for v in l[:3]])
                    self.scale_xyz = np.array([float(v) for v in l[3:]])
                print('{} exists'.format(save_fn))
                return

        all_xyzs = []
        for pid in self.points3D.keys():
            p3d = self.points3D[pid]
            obs = len(p3d.point2D_idxs)
            if obs < min_obs:
                continue
            all_xyzs.append(p3d.xyz)

        all_xyzs = np.array(all_xyzs)
        mean_xyz = np.ceil(np.mean(all_xyzs, axis=0))
        all_xyz_ = all_xyzs - mean_xyz

        dx = np.max(abs(all_xyz_[:, 0]))
        dy = np.max(abs(all_xyz_[:, 1]))
        dz = np.max(abs(all_xyz_[:, 2]))
        scale_xyz = np.ceil(np.array([dx, dy, dz], dtype=float).reshape(3, ))
        scale_xyz[scale_xyz < 1] = 1
        scale_xyz[scale_xyz == 0] = 1

        # self.mean_xyz = mean_xyz
        # self.scale_xyz = scale_xyz
        #
        # if save_fn is not None:
        #     with open(save_fn, 'w') as f:
        #         text = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(mean_xyz[0], mean_xyz[1], mean_xyz[2],
        #                                                                   scale_xyz[0], scale_xyz[1], scale_xyz[2])
        #         f.write(text + '\n')

    def compute_statics_inlier(self, xyz, nb_neighbors=20, std_ratio=2.0):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        new_pcd, inlier_ids = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return inlier_ids

    def export_features_to_directory(self, feat_fn, save_dir, with_descriptors=True):
        def print_grp_name(grp_name, object):
            try:
                n_subgroups = len(object.keys())
            except:
                n_subgroups = 0
                dataset_list.append(object.name)

        dataset_list = []
        feat_file = h5py.File(feat_fn, 'r')
        feat_file.visititems(print_grp_name)
        all_keys = []
        os.makedirs(save_dir, exist_ok=True)
        for fn in dataset_list:
            subs = fn[1:].split('/')[:-1]  # remove the first '/'
            subs = '/'.join(map(str, subs))
            if subs in all_keys:
                continue
            all_keys.append(subs)

        for fn in tqdm(all_keys, total=len(all_keys)):
            feat = feat_file[fn]
            data = {
                # 'descriptors': feat['descriptors'][()].transpose(),
                'scores': feat['scores'][()],
                'keypoints': feat['keypoints'][()],
                'image_size': feat['image_size'][()]
            }
            np.save(osp.join(save_dir, fn.replace('/', '+')), data)
        feat_file.close()

    def get_intrinsics_from_camera(self, camera):
        if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = camera.params[0]
            cx = camera.params[1]
            cy = camera.params[2]
        elif camera.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx = camera.params[0]
            fy = camera.params[1]
            cx = camera.params[2]
            cy = camera.params[3]
        else:
            raise Exception("Camera model not supported")

        # intrinsics
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def compress_map_by_projection_v2(self, vrf_path, point3d_desc_path, vrf_frames=1, covisible_frames=20, radius=20,
                                      nkpts=-1, save_dir=None):
        def sparsify_by_grid(h, w, uvs, scores):
            nh = np.ceil(h / radius).astype(int)
            nw = np.ceil(w / radius).astype(int)
            grid = {}
            for ip in range(uvs.shape[0]):
                p = uvs[ip]
                iw = np.rint(p[0] // radius).astype(int)
                ih = np.rint(p[1] // radius).astype(int)
                idx = ih * nw + iw
                if idx in grid.keys():
                    if scores[ip] <= grid[idx]['score']:
                        continue
                    else:
                        grid[idx]['score'] = scores[ip]
                        grid[idx]['ip'] = ip
                else:
                    grid[idx] = {
                        'score': scores[ip],
                        'ip': ip
                    }

            retained_ips = [grid[v]['ip'] for v in grid.keys()]
            retained_ips = np.array(retained_ips)
            return retained_ips

        def choose_valid_p3ds(current_frame_id, covisible_frame_ids, reserved_images):
            curr_p3d_ids = []
            curr_xyzs = []
            for pid in self.images[current_frame_id].point3D_ids:
                if pid == -1:
                    continue
                if pid not in self.points3D.keys():
                    continue
                curr_p3d_ids.append(pid)
                curr_xyzs.append(self.points3D[pid].xyz)
            curr_xyzs = np.array(curr_xyzs)  # [N, 3]
            curr_xyzs_homo = np.hstack([curr_xyzs, np.ones((curr_xyzs.shape[0], 1), dtype=curr_xyzs.dtype)])  # [N, 4]

            curr_mask = np.array([True for mi in range(curr_xyzs.shape[0])])  # keep all at first
            for iim in covisible_frame_ids:
                cam_id = self.images[iim].camera_id
                width = self.cameras[cam_id].width
                height = self.cameras[cam_id].height
                qvec = self.images[iim].qvec
                tcw = self.images[iim].tvec
                Rcw = qvec2rotmat(qvec=qvec)
                Tcw = np.eye(4, dtype=float)
                Tcw[:3, :3] = Rcw
                Tcw[:3, 3] = tcw.reshape(3, )

                uvs = reserved_images[iim]['xys']
                K = self.get_intrinsics_from_camera(camera=self.cameras[cam_id])
                proj_xys = K @ (Tcw @ curr_xyzs_homo.transpose())[:3, :]  # [3, ]
                proj_xys = proj_xys.transpose()
                depth = proj_xys[:, 2]
                proj_xys[:, 0] = proj_xys[:, 0] / depth
                proj_xys[:, 1] = proj_xys[:, 1] / depth

                mask_in_image = (proj_xys[:, 0] >= 0) * (proj_xys[:, 0] < width) * (proj_xys[:, 1] >= 0) * (
                        proj_xys[:, 1] < height)
                mask_depth = proj_xys[:, 2] > 0

                dist_proj_uv = torch.from_numpy(proj_xys[:, :2])[..., None] - \
                               torch.from_numpy(uvs[:, :2].transpose())[None]
                dist_proj_uv = torch.sqrt(torch.sum(dist_proj_uv ** 2, dim=1))  # [N, M]
                min_dist = torch.min(dist_proj_uv, dim=1)[0].numpy()
                mask_close_to_img = (min_dist <= radius)

                mask = mask_in_image * mask_depth * mask_close_to_img  # p3ds to be discarded

                curr_mask = curr_mask * (1 - mask)

            chosen_p3d_ids = []
            for mi in range(curr_mask.shape[0]):
                if curr_mask[mi]:
                    chosen_p3d_ids.append(curr_p3d_ids[mi])

            return chosen_p3d_ids

        vrf_data = np.load(vrf_path, allow_pickle=True)[()]
        p3d_ids_in_vrf = []
        image_ids_in_vrf = []
        for sid in vrf_data.keys():
            svrf = vrf_data[sid]
            svrf_keys = [vi for vi in range(vrf_frames)]
            for vi in svrf_keys:
                if vi not in svrf.keys():
                    continue
                image_id = svrf[vi]['image_id']
                if image_id in image_ids_in_vrf:
                    continue
                image_ids_in_vrf.append(image_id)
                for pid in svrf[vi]['original_points3d']:
                    if pid in p3d_ids_in_vrf:
                        continue
                    p3d_ids_in_vrf.append(pid)

        print('Find {:d} images and {:d} 3D points in vrf'.format(len(image_ids_in_vrf), len(p3d_ids_in_vrf)))

        # first_vrf_images_covis = {}
        retained_image_ids = {}
        for frame_id in image_ids_in_vrf:
            observed = self.images[frame_id].point3D_ids
            xys = self.images[frame_id].xys
            covis = defaultdict(int)
            valid_xys = []
            valid_p3d_ids = []
            for xy, pid in zip(xys, observed):
                if pid == -1:
                    continue
                if pid not in self.points3D.keys():
                    continue
                valid_xys.append(xy)
                valid_p3d_ids.append(pid)
                for img_id in self.points3D[pid].image_ids:
                    covis[img_id] += 1

            retained_image_ids[frame_id] = {
                'xys': np.array(valid_xys),
                'p3d_ids': valid_p3d_ids,
            }

            print('Find {:d} valid connected frames'.format(len(covis.keys())))

            covis_ids = np.array(list(covis.keys()))
            covis_num = np.array([covis[i] for i in covis_ids])

            if len(covis_ids) <= covisible_frames:
                sel_covis_ids = covis_ids[np.argsort(-covis_num)]
            else:
                ind_top = np.argpartition(covis_num, -covisible_frames)
                ind_top = ind_top[-covisible_frames:]  # unsorted top k
                ind_top = ind_top[np.argsort(-covis_num[ind_top])]
                sel_covis_ids = [covis_ids[i] for i in ind_top]

            covis_frame_ids = [frame_id]
            for iim in sel_covis_ids:
                if iim == frame_id:
                    continue
                if iim in retained_image_ids.keys():
                    covis_frame_ids.append(iim)
                    continue

                chosen_p3d_ids = choose_valid_p3ds(current_frame_id=iim, covisible_frame_ids=covis_frame_ids,
                                                   reserved_images=retained_image_ids)
                if len(chosen_p3d_ids) == 0:
                    continue

                xys = []
                for xy, pid in zip(self.images[iim].xys, self.images[iim].point3D_ids):
                    if pid in chosen_p3d_ids:
                        xys.append(xy)
                xys = np.array(xys)

                covis_frame_ids.append(iim)
                retained_image_ids[iim] = {
                    'xys': xys,
                    'p3d_ids': chosen_p3d_ids,
                }

        new_images = {}
        new_point3Ds = {}
        new_cameras = {}
        for iim in retained_image_ids.keys():
            p3d_ids = retained_image_ids[iim]['p3d_ids']
            ''' this step reduces the performance
            for v in self.images[iim].point3D_ids:
                if v == -1 or v not in self.points3D:
                    continue
                if v in p3d_ids:
                    continue
                p3d_ids.append(v)
            '''

            xyzs = np.array([self.points3D[pid].xyz for pid in p3d_ids])
            obs = np.array([len(self.points3D[pid].point2D_idxs) for pid in p3d_ids])
            xys = self.images[iim].xys
            cam_id = self.images[iim].camera_id
            name = self.images[iim].name
            qvec = self.images[iim].qvec
            tvec = self.images[iim].tvec

            if nkpts > 0 and len(p3d_ids) > nkpts:
                proj_uvs = self.reproject(img_id=iim, xyzs=xyzs)
                width = self.cameras[cam_id].width
                height = self.cameras[cam_id].height
                sparsified_idxs = sparsify_by_grid(h=height, w=width, uvs=proj_uvs[:, :2], scores=obs)

                print('org / new kpts: ', len(p3d_ids), sparsified_idxs.shape)

                p3d_ids = [p3d_ids[k] for k in sparsified_idxs]

            new_images[iim] = Image(id=iim, qvec=qvec, tvec=tvec,
                                    camera_id=cam_id,
                                    name=name,
                                    xys=np.array([]),
                                    point3D_ids=np.array(p3d_ids))

            if cam_id not in new_cameras.keys():
                new_cameras[cam_id] = self.cameras[cam_id]

            for pid in p3d_ids:
                if pid in new_point3Ds.keys():
                    new_point3Ds[pid]['image_ids'].append(iim)
                else:
                    xyz = self.points3D[pid].xyz
                    rgb = self.points3D[pid].rgb
                    error = self.points3D[pid].error

                    new_point3Ds[pid] = {
                        'image_ids': [iim],
                        'rgb': rgb,
                        'xyz': xyz,
                        'error': error
                    }

        new_point3Ds_to_save = {}
        for pid in new_point3Ds.keys():
            image_ids = new_point3Ds[pid]['image_ids']
            if len(image_ids) == 0:
                continue
            xyz = new_point3Ds[pid]['xyz']
            rgb = new_point3Ds[pid]['rgb']
            error = new_point3Ds[pid]['error']

            new_point3Ds_to_save[pid] = Point3D(id=pid, xyz=xyz, rgb=rgb, error=error, image_ids=np.array(image_ids),
                                                point2D_idxs=np.array([]))

        print('Retain {:d}/{:d} images and {:d}/{:d} 3D points'.format(len(new_images), len(self.images),
                                                                       len(new_point3Ds), len(self.points3D)))

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # write_images_binary(images=new_image_ids,
            #                     path_to_model_file=osp.join(save_dir, 'images.bin'))
            # write_points3d_binary(points3D=new_point3Ds,
            #                       path_to_model_file=osp.join(save_dir, 'points3D.bin'))
            write_compressed_images_binary(images=new_images,
                                           path_to_model_file=osp.join(save_dir, 'images.bin'))
            write_cameras_binary(cameras=new_cameras,
                                 path_to_model_file=osp.join(save_dir, 'cameras.bin'))
            write_compressed_points3d_binary(points3D=new_point3Ds_to_save,
                                             path_to_model_file=osp.join(save_dir, 'points3D.bin'))

            # Save 3d descriptors
            p3d_desc = np.load(point3d_desc_path, allow_pickle=True)[()]
            comp_p3d_desc = {}
            for k in new_point3Ds_to_save.keys():
                if k not in p3d_desc.keys():
                    print(k)
                    continue
                comp_p3d_desc[k] = deepcopy(p3d_desc[k])
            np.save(osp.join(save_dir, point3d_desc_path.split('/')[-1]), comp_p3d_desc)
            print('Save data to {:s}'.format(save_dir))


def process_dataset():
    dataset_dir = '/scratches/flyer_3/fx221/dataset'
    sfm_dir = '/scratches/flyer_2/fx221/localization/outputs'  # your sfm results (cameras, images, points3D) and features
    save_dir = '/scratches/flyer_3/fx221/exp/localizer'
    hloc_results_dir = '/scratches/flyer_2/fx221/exp/sgd2'

    local_feat = 'sfd2'
    matcher = 'gml'

    # config_path = 'configs/datasets/CUED.yaml'
    config_path = 'configs/datasets/7Scenes.yaml'
    # config_path = 'configs/datasets/12Scenes.yaml'
    # config_path = 'configs/datasets/CambridgeLandmarks.yaml'
    # config_path = 'configs/datasets/Aachen.yaml'

    # config_path = 'configs/datasets/Aria.yaml'
    # config_path = 'configs/datasets/DarwinRGB.yaml'
    # config_path = 'configs/datasets/ACUED.yaml'
    # config_path = 'configs/datasets/JesusCollege.yaml'
    # config_path = 'configs/datasets/CUED2Kings.yaml'

    with open(config_path, 'rt') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    print(configs)

    dataset = configs['dataset']
    all_scenes = configs['scenes']
    for scene in all_scenes:
        n_cluster = configs[scene]['n_cluster']
        cluster_mode = configs[scene]['cluster_mode']
        cluster_method = configs[scene]['cluster_method']

        print('scene: ', scene, cluster_mode, cluster_method)

        # hloc_path = osp.join(hloc_root, dataset, scene)
        sfm_path = osp.join(sfm_dir, dataset, scene)
        feat_path = osp.join(sfm_dir, dataset, scene, 'feats-{:s}.h5'.format(local_feat))
        save_path = osp.join(save_dir, local_feat + '-' + matcher, dataset, scene)

        n_vrf = 1
        n_cov = 30
        radius = 20
        n_kpts = 0

        if dataset in ['Aachen']:
            image_path = osp.join(dataset_dir, dataset, scene, 'images/images_upright')
            min_obs = 250
            filtering_outliers = True
            threshold = 0.2
            radius = 32

        elif dataset in ['CambridgeLandmarks', ]:
            image_path = osp.join(dataset_dir, dataset, scene)
            min_obs = 250
            filtering_outliers = True
            threshold = 0.2
            radius = 64
        elif dataset in ['Aria']:
            image_path = osp.join(dataset_dir, dataset, scene)
            min_obs = 150
            filtering_outliers = False
            threshold = 0.01
            radius = 15
        elif dataset in ['DarwinRGB']:
            image_path = osp.join(dataset_dir, dataset, scene)
            min_obs = 150
            filtering_outliers = True
            threshold = 0.2
            radius = 16
        elif dataset in ['ACUED']:
            image_path = osp.join(dataset_dir, dataset, scene)
            min_obs = 250
            filtering_outliers = True
            threshold = 0.2
            radius = 32
        elif dataset in ['7Scenes', '12Scenes']:
            image_path = osp.join(dataset_dir, dataset, scene)
            min_obs = 150
            filtering_outliers = False
            threshold = 0.01
            radius = 15
        else:
            image_path = osp.join(dataset_dir, dataset, scene)
            min_obs = 250
            filtering_outliers = True
            threshold = 0.2
            radius = 32

        # comp_map_sub_path = 'comp_model_n{:d}_{:s}_{:s}_vrf{:d}_cov{:d}_r{:d}_np{:d}_projection_v2'.format(n_cluster,
        #                                                                                                    cluster_mode,
        #                                                                                                    cluster_method,
        #                                                                                                    n_vrf,
        #                                                                                                    n_cov,
        #                                                                                                    radius,
        #                                                                                                    n_kpts)
        comp_map_sub_path = 'compress_model_{:s}'.format(cluster_method)
        seg_fn = osp.join(save_path,
                          'point3D_cluster_n{:d}_{:s}_{:s}.npy'.format(n_cluster, cluster_mode, cluster_method))
        vrf_fn = osp.join(save_path,
                          'point3D_vrf_n{:d}_{:s}_{:s}.npy'.format(n_cluster, cluster_mode, cluster_method))
        vrf_img_dir = osp.join(save_path,
                               'point3D_vrf_n{:d}_{:s}_{:s}'.format(n_cluster, cluster_mode, cluster_method))
        # p3d_query_fn = osp.join(save_path,
        #                         'point3D_query_n{:d}_{:s}_{:s}.npy'.format(n_cluster, cluster_mode, cluster_method))
        comp_map_path = osp.join(save_path, comp_map_sub_path)

        os.makedirs(save_path, exist_ok=True)

        rmap = RecMap()
        rmap.load_sfm_model(path=osp.join(sfm_path, 'sfm_{:s}-{:s}'.format(local_feat, matcher)))
        if filtering_outliers:
            rmap.remove_statics_outlier(nb_neighbors=20, std_ratio=2.0)

        # extract keypoints to train the recognition model (descriptors are recomputed from augmented db images)
        rmap.export_features_to_directory(feat_fn=osp.join(sfm_path, 'feats-{:s}.h5'.format(local_feat)),
                                          save_dir=osp.join(save_path, 'feats'))  # only once for training

        rmap.cluster(k=n_cluster, mode=cluster_mode, save_fn=seg_fn, method=cluster_method, threshold=threshold)
        rmap.visualize_3Dpoints()
        rmap.load_segmentation(path=seg_fn)
        rmap.visualize_segmentation(p3d_segs=rmap.p3d_seg, points3D=rmap.points3D)
        # rmap.compute_mean_scale_p3ds(min_obs=5, save_fn=osp.join(save_path, 'sc_mean_scale.txt'))

        # Assign each 3D point a desciptor and discard all 2D images and descriptors - for localization
        rmap.assign_point3D_descriptor(
            feature_fn=osp.join(sfm_path, 'feats-{:s}.h5'.format(local_feat)),
            save_fn=osp.join(save_path, 'point3D_desc.npy'.format(n_cluster, cluster_mode)),
            n_process=32)  # only once

        # exit(0)
        # rmap.visualize_segmentation_on_image(p3d_segs=rmap.p3d_seg, image_path=image_path, feat_path=feat_path)

        # for query images only - for evaluation
        # rmap.extract_query_p3ds(
        #     log_fn=osp.join(hloc_path, 'hloc_feats-{:s}_{:s}_loc.npy'.format(local_feat, matcher)),
        #     feat_fn=osp.join(sfm_path, 'feats-{:s}.h5'.format(local_feat)),
        #     save_fn=p3d_query_fn)
        # continue

        # up-to-date
        rmap.create_virtual_frame_3(
            save_fn=vrf_fn,
            save_vrf_dir=vrf_img_dir,
            image_root=image_path,
            show_time=5,
            min_cover_ratio=0.9,
            radius=radius,
            depth_scale=2.5,  # 1.2 by default
            min_obs=min_obs,
            n_vrf=10,
            covisible_frame=n_cov,
            ignored_cameras=[])

        # up-to-date
        rmap.compress_map_by_projection_v2(
            vrf_frames=n_vrf,
            vrf_path=vrf_fn,
            point3d_desc_path=osp.join(save_path, 'point3D_desc.npy'),
            save_dir=comp_map_path,
            covisible_frames=n_cov,
            radius=radius,
            nkpts=n_kpts,
        )

        # exit(0)
        # soft_link_compress_path = osp.join(save_path, 'compress_model_{:s}'.format(cluster_method))
        os.chdir(save_path)
        # if osp.isdir(soft_link_compress_path):
        #     os.unlink(soft_link_compress_path)
        # os.symlink(comp_map_sub_path, 'compress_model_{:s}'.format(cluster_method))
        # create a soft link for full model
        if not osp.isdir('model'):
            os.symlink(osp.join(sfm_path, 'sfm_{:s}-{:s}'.format(local_feat, matcher)), 'model')


if __name__ == '__main__':
    process_dataset()
