# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> twelve_scenes
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:37
=================================================='''
import os
import os.path as osp
import numpy as np
from colmap_utils.read_write_model import read_model
import torchvision.transforms as tvt
from dataset.basicdataset import BasicDataset


class TwelveScenes(BasicDataset):
    def __init__(self, landmark_path, scene, dataset_path, n_class, seg_mode, seg_method, dataset='12Scenes',
                 nfeatures=1024,
                 query_p3d_fn=None,
                 train=True,
                 with_aug=False,
                 min_inliers=0,
                 max_inliers=4096,
                 random_inliers=False,
                 jitter_params=None,
                 scale_params=None,
                 image_dim=3,
                 query_info_path=None,
                 sample_ratio=1,
                 ):
        self.landmark_path = osp.join(landmark_path, scene)
        self.dataset_path = osp.join(dataset_path, scene)
        self.n_class = n_class
        self.dataset = dataset + '/' + scene
        self.nfeatures = nfeatures
        self.with_aug = with_aug
        self.jitter_params = jitter_params
        self.scale_params = scale_params
        self.image_dim = image_dim
        self.train = train
        self.min_inliers = min_inliers
        self.max_inliers = max_inliers if max_inliers < nfeatures else nfeatures
        self.random_inliers = random_inliers
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

        if train:
            self.cameras, self.images, point3Ds = read_model(path=osp.join(self.landmark_path, '3D-models'), ext='.bin')
            self.name_to_id = {image.name: i for i, image in self.images.items() if len(self.images[i].point3D_ids) > 0}

        # only for testing of query images
        if not self.train:
            data = np.load(query_p3d_fn, allow_pickle=True)[()]
            self.img_p3d = data
        else:
            self.img_p3d = {}

        with open(osp.join(self.dataset_path, 'split.txt'), 'r') as f:
            l = f.readline()
            l = l.strip().split(' ')  # sequence0 [frames=357]  [start=0 ; end=356],  first sequence for testing
            start_img_id = l[-3].split('=')[-1]
            end_img_id = l[-1].split('=')[-1][:-1]
            test_start_img_id = int(start_img_id)
            test_end_img_id = int(end_img_id)

        self.img_fns = []
        fns = os.listdir(osp.join(self.dataset_path, 'data'))
        fns = sorted(fns)
        nf = 0
        for fn in fns:
            if fn.find('jpg') >= 0:  # frame-001098.color.jpg
                frame_id = int(fn.split('.')[0].split('-')[-1])
                if not train and frame_id > test_end_img_id:
                    continue
                if train and frame_id <= test_end_img_id:
                    continue

                if train and 'data' + '/' + fn not in self.name_to_id.keys():
                    continue

                if not train and 'data' + '/' + fn not in self.img_p3d.keys():
                    continue
                if nf % sample_ratio == 0:
                    self.img_fns.append('data' + '/' + fn)
                nf += 1

        print('Load {} images from {} for {}...'.format(len(self.img_fns),
                                                        self.dataset, 'training' if train else 'eval'))

        data = np.load(osp.join(self.landmark_path,
                                'point3D_cluster_n{:d}_{:s}_{:s}.npy'.format(n_class - 1, seg_mode, seg_method)),
                       allow_pickle=True)[()]
        p3d_id = data['id']
        seg_id = data['label']
        self.p3d_seg = {p3d_id[i]: seg_id[i] for i in range(p3d_id.shape[0])}
        xyzs = data['xyz']
        self.p3d_xyzs = {p3d_id[i]: xyzs[i] for i in range(p3d_id.shape[0])}

        # with open(osp.join(self.landmark_path, 'sc_mean_scale.txt'), 'r') as f:
        #     lines = f.readlines()
        #     for l in lines:
        #         l = l.strip().split()
        #         self.mean_xyz = np.array([float(v) for v in l[:3]])
        #         self.scale_xyz = np.array([float(v) for v in l[3:]])

        if not train:
            self.query_info = self.read_query_info(path=query_info_path)

        self.nfeatures = nfeatures
        self.feature_dir = osp.join(self.landmark_path, 'feats')
        self.feats = {}
