# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> basicdataset
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:27
=================================================='''
import torchvision.transforms.functional as tvf
import torchvision.transforms as tvt
import os.path as osp
import numpy as np
import cv2
from colmap_utils.read_write_model import qvec2rotmat, read_model
from dataset.utils import normalize_size


class BasicDataset:
    def __init__(self,
                 img_list_fn,
                 feature_dir,
                 sfm_path,
                 seg_fn,
                 dataset_path,
                 n_class,
                 dataset,
                 nfeatures=1024,
                 query_p3d_fn=None,
                 train=True,
                 with_aug=False,
                 min_inliers=0,
                 max_inliers=4096,
                 random_inliers=False,
                 jitter_params=None,
                 scale_params=None,
                 image_dim=1,
                 pre_load=False,
                 query_info_path=None,
                 sc_mean_scale_fn=None,
                 ):
        self.n_class = n_class
        self.train = train
        self.min_inliers = min_inliers
        self.max_inliers = max_inliers if max_inliers < nfeatures else nfeatures
        self.random_inliers = random_inliers
        self.dataset_path = dataset_path
        self.with_aug = with_aug
        self.dataset = dataset
        self.jitter_params = jitter_params
        self.scale_params = scale_params
        self.image_dim = image_dim
        self.image_prefix = ''

        train_transforms = []
        if self.with_aug:
            train_transforms.append(tvt.ColorJitter(
                brightness=jitter_params['brightness'],
                contrast=jitter_params['contrast'],
                saturation=jitter_params['saturation'],
                hue=jitter_params['hue']))
            if jitter_params['blur'] > 0:
                train_transforms.append(tvt.GaussianBlur(kernel_size=int(jitter_params['blur'])))
        self.train_transforms = tvt.Compose(train_transforms)

        # only for testing of query images
        if not self.train:
            data = np.load(query_p3d_fn, allow_pickle=True)[()]
            self.img_p3d = data
        else:
            self.img_p3d = {}

        self.img_fns = []
        with open(img_list_fn, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                self.img_fns.append(l)
        print('Load {} images from {} for {}...'.format(len(self.img_fns), dataset, 'training' if train else 'eval'))
        self.feats = {}
        if train:
            self.cameras, self.images, point3Ds = read_model(path=sfm_path, ext='.bin')
            self.name_to_id = {image.name: i for i, image in self.images.items()}

        data = np.load(seg_fn, allow_pickle=True)[()]
        p3d_id = data['id']
        seg_id = data['label']
        self.p3d_seg = {p3d_id[i]: seg_id[i] for i in range(p3d_id.shape[0])}
        self.p3d_xyzs = {}

        for pid in self.p3d_seg.keys():
            p3d = point3Ds[pid]
            self.p3d_xyzs[pid] = p3d.xyz

        with open(sc_mean_scale_fn, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split()
                self.mean_xyz = np.array([float(v) for v in l[:3]])
                self.scale_xyz = np.array([float(v) for v in l[3:]])

        if not train:
            self.query_info = self.read_query_info(path=query_info_path)

        self.nfeatures = nfeatures
        self.feature_dir = feature_dir
        print('Pre loaded {} feats, mean xyz {}, scale xyz {}'.format(len(self.feats.keys()), self.mean_xyz,
                                                                      self.scale_xyz))

    def normalize_p3ds(self, p3ds):
        mean_p3ds = np.ceil(np.mean(p3ds, axis=0))
        p3ds_ = p3ds - mean_p3ds
        dx = np.max(abs(p3ds_[:, 0]))
        dy = np.max(abs(p3ds_[:, 1]))
        dz = np.max(abs(p3ds_[:, 2]))
        scale_p3ds = np.ceil(np.array([dx, dy, dz], dtype=float).reshape(3, ))
        scale_p3ds[scale_p3ds < 1] = 1
        scale_p3ds[scale_p3ds == 0] = 1
        return mean_p3ds, scale_p3ds

    def read_query_info(self, path):
        query_info = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split()
                image_name = l[0]
                cam_model = l[1]
                h, w = int(l[2]), int(l[3])
                params = np.array([float(v) for v in l[4:]])
                query_info[image_name] = {
                    'width': w,
                    'height': h,
                    'model': cam_model,
                    'params': params,
                }
        return query_info

    def extract_intrinsic_extrinsic_params(self, image_id):
        cam = self.cameras[self.images[image_id].camera_id]
        params = cam.params
        model = cam.model
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
        K = np.eye(3, dtype=float)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        qvec = self.images[image_id].qvec
        tvec = self.images[image_id].tvec
        R = qvec2rotmat(qvec=qvec)
        P = np.eye(4, dtype=float)
        P[:3, :3] = R
        P[:3, 3] = tvec.reshape(3, )

        return {'K': K, 'P': P}

    def get_item_train(self, idx):
        img_name = self.img_fns[idx]
        if img_name in self.feats.keys():
            feat_data = self.feats[img_name]
        else:
            feat_data = np.load(osp.join(self.feature_dir, img_name.replace('/', '+') + '.npy'), allow_pickle=True)[()]
        # descs = feat_data['descriptors']  # [N, D]
        scores = feat_data['scores']  # [N, 1]
        kpts = feat_data['keypoints']  # [N, 2]
        image_size = feat_data['image_size']

        nfeat = kpts.shape[0]

        # print(img_name, self.name_to_id[img_name])
        p3d_ids = self.images[self.name_to_id[img_name]].point3D_ids
        p3d_xyzs = np.zeros(shape=(nfeat, 3), dtype=float)

        seg_ids = np.zeros(shape=(nfeat,), dtype=int)  # + self.n_class - 1
        for i in range(nfeat):
            p3d = p3d_ids[i]
            if p3d in self.p3d_seg.keys():
                seg_ids[i] = self.p3d_seg[p3d] + 1  # 0 for invalid
                if seg_ids[i] == -1:
                    seg_ids[i] = 0

            if p3d in self.p3d_xyzs.keys():
                p3d_xyzs[i] = self.p3d_xyzs[p3d]

        seg_ids = np.array(seg_ids).reshape(-1, )

        n_inliers = np.sum(seg_ids > 0)
        n_outliers = np.sum(seg_ids == 0)
        inlier_ids = np.where(seg_ids > 0)[0]
        outlier_ids = np.where(seg_ids == 0)[0]

        if n_inliers <= self.min_inliers:
            sel_inliers = n_inliers
            sel_outliers = self.nfeatures - sel_inliers

            out_ids = np.arange(n_outliers)
            np.random.shuffle(out_ids)
            sel_ids = np.hstack([inlier_ids, outlier_ids[out_ids[:self.nfeatures - n_inliers]]])
        else:
            sel_inliers = np.random.randint(self.min_inliers, self.max_inliers)
            if sel_inliers > n_inliers:
                sel_inliers = n_inliers

            if sel_inliers + n_outliers < self.nfeatures:
                sel_inliers = self.nfeatures - n_outliers

            sel_outliers = self.nfeatures - sel_inliers

            in_ids = np.arange(n_inliers)
            np.random.shuffle(in_ids)
            sel_inlier_ids = inlier_ids[in_ids[:sel_inliers]]

            out_ids = np.arange(n_outliers)
            np.random.shuffle(out_ids)
            sel_outlier_ids = outlier_ids[out_ids[:sel_outliers]]

            sel_ids = np.hstack([sel_inlier_ids, sel_outlier_ids])

        # sel_descs = descs[sel_ids]
        sel_scores = scores[sel_ids]
        sel_kpts = kpts[sel_ids]
        sel_seg_ids = seg_ids[sel_ids]
        sel_xyzs = p3d_xyzs[sel_ids]

        shuffle_ids = np.arange(sel_ids.shape[0])
        np.random.shuffle(shuffle_ids)
        # sel_descs = sel_descs[shuffle_ids]
        sel_scores = sel_scores[shuffle_ids]
        sel_kpts = sel_kpts[shuffle_ids]
        sel_seg_ids = sel_seg_ids[shuffle_ids]
        sel_xyzs = sel_xyzs[shuffle_ids]

        if sel_kpts.shape[0] < self.nfeatures:
            # print(sel_descs.shape, sel_kpts.shape, sel_scores.shape, sel_seg_ids.shape, sel_xyzs.shape)
            valid_sel_ids = np.array([v for v in range(sel_kpts.shape[0]) if sel_seg_ids[v] > 0], dtype=int)
            # ref_sel_id = np.random.choice(valid_sel_ids, size=1)[0]
            if valid_sel_ids.shape[0] == 0:
                valid_sel_ids = np.array([v for v in range(sel_kpts.shape[0])], dtype=int)
            random_n = self.nfeatures - sel_kpts.shape[0]
            random_scores = np.random.random((random_n,))
            random_kpts, random_seg_ids, random_xyzs = self.random_points_from_reference(
                n=random_n,
                ref_kpts=sel_kpts[valid_sel_ids],
                ref_segs=sel_seg_ids[valid_sel_ids],
                ref_xyzs=sel_xyzs[valid_sel_ids],
                radius=5,
            )
            # sel_descs = np.vstack([sel_descs, random_descs])
            sel_scores = np.hstack([sel_scores, random_scores])
            sel_kpts = np.vstack([sel_kpts, random_kpts])
            sel_seg_ids = np.hstack([sel_seg_ids, random_seg_ids])
            sel_xyzs = np.vstack([sel_xyzs, random_xyzs])

        gt_n_seg = np.zeros(shape=(self.n_class,), dtype=int)
        gt_cls = np.zeros(shape=(self.n_class,), dtype=int)
        gt_cls_dist = np.zeros(shape=(self.n_class,), dtype=float)
        uids = np.unique(sel_seg_ids).tolist()
        for uid in uids:
            if uid == 0:
                continue
            gt_cls[uid] = 1
            gt_n_seg[uid] = np.sum(sel_seg_ids == uid)
            gt_cls_dist[uid] = np.sum(seg_ids == uid) / np.sum(seg_ids > 0)  # [valid_id / total_valid_id]

        param_out = self.extract_intrinsic_extrinsic_params(image_id=self.name_to_id[img_name])
        output = {
            # 'descriptors': sel_descs,  # may not be used
            'scores': sel_scores,
            'keypoints': sel_kpts,
            'norm_keypoints': normalize_size(x=sel_kpts, size=image_size),
            'gt_seg': sel_seg_ids,
            'gt_cls': gt_cls,
            'gt_cls_dist': gt_cls_dist,
            'gt_n_seg': gt_n_seg,
            'file_name': img_name,
            'prefix_name': self.image_prefix,
            'mean_xyz': self.mean_xyz,
            'scale_xyz': self.scale_xyz,
            'gt_sc': sel_xyzs,
            'gt_norm_sc': (sel_xyzs - self.mean_xyz) / self.scale_xyz,
            'K': param_out['K'],
            'gt_P': param_out['P']
        }

        img = self.read_image(image_name=img_name)
        if self.image_dim == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.with_aug:
            nh = img.shape[0]
            nw = img.shape[1]
            if self.scale_params is not None:
                do_scale = np.random.random()
                if do_scale <= 0.25:
                    p = np.random.randint(0, 11)
                    s = self.scale_params[0] + (self.scale_params[1] - self.scale_params[0]) / 10 * p
                    nh = int(img.shape[0] * s)
                    nw = int(img.shape[1] * s)
                    sh = nh / img.shape[0]
                    sw = nw / img.shape[1]
                    sel_kpts[:, 0] = sel_kpts[:, 0] * sw
                    sel_kpts[:, 1] = sel_kpts[:, 1] * sh
                    img = cv2.resize(img, dsize=(nw, nh))

            brightness = np.random.uniform(-self.jitter_params['brightness'], self.jitter_params['brightness']) * 255
            contrast = 1 + np.random.uniform(-self.jitter_params['contrast'], self.jitter_params['contrast'])
            img = cv2.addWeighted(img, contrast, img, 0, brightness)
            img = np.clip(img, a_min=0, a_max=255)
            if self.image_dim == 1:
                img = img[..., None]
            output['image'] = [img.astype(float) / 255.]
            output['image_size'] = np.array([nh, nw], dtype=int)
        else:
            if self.image_dim == 1:
                img = img[..., None].astype(float) / 255.
            output['image'] = [img]

        return output

    def get_item_test(self, idx):

        # evaluation of recognition only
        img_name = self.img_fns[idx]
        feat_data = np.load(osp.join(self.feature_dir, img_name.replace('/', '+') + '.npy'), allow_pickle=True)[()]
        descs = feat_data['descriptors']  # [N, D]
        scores = feat_data['scores']  # [N, 1]
        kpts = feat_data['keypoints']  # [N, 2]
        image_size = feat_data['image_size']

        nfeat = descs.shape[0]

        if img_name in self.img_p3d.keys():
            p3d_ids = self.img_p3d[img_name]
        p3d_xyzs = np.zeros(shape=(nfeat, 3), dtype=float)
        seg_ids = np.zeros(shape=(nfeat,), dtype=int)  # attention! by default invalid!!!
        for i in range(nfeat):
            p3d = p3d_ids[i]
            if p3d in self.p3d_seg.keys():
                seg_ids[i] = self.p3d_seg[p3d] + 1
                if seg_ids[i] == -1:
                    seg_ids[i] = 0  # 0  for in valid

            if p3d in self.p3d_xyzs.keys():
                p3d_xyzs[i] = self.p3d_xyzs[p3d]

        seg_ids = np.array(seg_ids).reshape(-1, )

        if self.nfeatures > 0:
            sorted_ids = np.argsort(scores)[::-1][:self.nfeatures]  # large to small
            descs = descs[sorted_ids]
            scores = scores[sorted_ids]
            kpts = kpts[sorted_ids]
            p3d_xyzs = p3d_xyzs[sorted_ids]

            seg_ids = seg_ids[sorted_ids]

        gt_n_seg = np.zeros(shape=(self.n_class,), dtype=int)
        gt_cls = np.zeros(shape=(self.n_class,), dtype=int)
        gt_cls_dist = np.zeros(shape=(self.n_class,), dtype=float)
        uids = np.unique(seg_ids).tolist()
        for uid in uids:
            if uid == 0:
                continue
            gt_cls[uid] = 1
            gt_n_seg[uid] = np.sum(seg_ids == uid)
            gt_cls_dist[uid] = np.sum(seg_ids == uid) / np.sum(
                seg_ids < self.n_class - 1)  # [valid_id / total_valid_id]

        gt_cls[0] = 0

        img = self.read_image(image_name=img_name)
        if self.image_dim == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[..., None].astype(float) / 255.
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.
        return {
            'descriptors': descs,
            'scores': scores,
            'keypoints': kpts,
            'image_size': image_size,
            'norm_keypoints': normalize_size(x=kpts, size=image_size),
            'gt_seg': seg_ids,
            'gt_cls': gt_cls,
            'gt_cls_dist': gt_cls_dist,
            'gt_n_seg': gt_n_seg,
            'file_name': img_name,
            'prefix_name': self.image_prefix,
            'image': [img],

            'mean_xyz': self.mean_xyz,
            'scale_xyz': self.scale_xyz,
            'gt_sc': p3d_xyzs,
            'gt_norm_sc': (p3d_xyzs - self.mean_xyz) / self.scale_xyz
        }

    def __getitem__(self, idx):
        if self.train:
            return self.get_item_train(idx=idx)
        else:
            return self.get_item_test(idx=idx)

    def __len__(self):
        return len(self.img_fns)

    def read_image(self, image_name):
        return cv2.imread(osp.join(self.dataset_path, image_name))

    def jitter_augmentation(self, img, params):
        brightness, contrast, saturation, hue = params
        p = np.random.randint(0, 20) / 20
        b = brightness[0] + (brightness[1] - brightness[0]) / 20 * p
        img = tvf.adjust_brightness(img=img, brightness_factor=b)

        p = np.random.randint(0, 20) / 20
        c = contrast[0] + (contrast[1] - contrast[0]) / 20 * p
        img = tvf.adjust_contrast(img=img, contrast_factor=c)

        p = np.random.randint(0, 20) / 20
        s = saturation[0] + (saturation[1] - saturation[0]) / 20 * p
        img = tvf.adjust_saturation(img=img, saturation_factor=s)

        p = np.random.randint(0, 20) / 20
        h = hue[0] + (hue[1] - hue[0]) / 20 * p
        img = tvf.adjust_hue(img=img, hue_factor=h)

        return img

    def random_points(self, n, d, h, w):
        desc = np.random.random((n, d))
        desc = desc / np.linalg.norm(desc, ord=2, axis=1)[..., None]
        xs = np.random.randint(0, w - 1, size=(n, 1))
        ys = np.random.randint(0, h - 1, size=(n, 1))
        kpts = np.hstack([xs, ys])
        return desc, kpts

    def random_points_from_reference(self, n, ref_kpts, ref_segs, ref_xyzs, radius=5):
        n_ref = ref_kpts.shape[0]
        if n_ref < n:
            ref_ids = np.random.choice([i for i in range(n_ref)], size=n).tolist()
        else:
            ref_ids = [i for i in range(n)]

        new_xs = []
        new_ys = []
        # new_descs = []
        new_segs = []
        new_xyzs = []
        for i in ref_ids:
            nx = np.random.randint(-radius, radius) + ref_kpts[i, 0]
            ny = np.random.randint(-radius, radius) + ref_kpts[i, 1]

            new_xs.append(nx)
            new_ys.append(ny)
            # new_descs.append(ref_descs[i])
            new_segs.append(ref_segs[i])
            new_xyzs.append(ref_xyzs[i])

        new_xs = np.array(new_xs).reshape(n, 1)
        new_ys = np.array(new_ys).reshape(n, 1)
        new_segs = np.array(new_segs).reshape(n, )
        new_kpts = np.hstack([new_xs, new_ys])
        # new_descs = np.array(new_descs).reshape(n, -1)
        new_xyzs = np.array(new_xyzs)
        return new_kpts, new_segs, new_xyzs
