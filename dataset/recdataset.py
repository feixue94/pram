# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> recdataset
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:42
=================================================='''
import numpy as np
from torch.utils.data import Dataset


class RecDataset(Dataset):
    def __init__(self, sub_sets=[]):
        assert len(sub_sets) >= 1

        self.sub_sets = sub_sets
        self.names = []

        self.sub_set_index = []
        self.seg_offsets = []
        self.sub_set_item_index = []
        self.dataset_names = []
        self.scene_names = []
        start_index_valid_seg = 1  # start from 1, 0 is for invalid

        total_subset = 0
        for scene_set in sub_sets:  # [0, n_class]
            name = scene_set.dataset
            self.names.append(name)
            n_samples = len(scene_set)

            n_class = scene_set.n_class
            self.seg_offsets = self.seg_offsets + [start_index_valid_seg for v in range(len(scene_set))]
            start_index_valid_seg = start_index_valid_seg + n_class - 1

            self.sub_set_index = self.sub_set_index + [total_subset for k in range(n_samples)]
            self.sub_set_item_index = self.sub_set_item_index + [k for k in range(n_samples)]

            # self.dataset_names = self.dataset_names + [name for k in range(n_samples)]
            self.scene_names = self.scene_names + [name for k in range(n_samples)]
            total_subset += 1

        self.n_class = start_index_valid_seg

        print('Load {} images {} segs from {} subsets from {}'.format(len(self.sub_set_item_index), self.n_class,
                                                                      len(sub_sets), self.names))

    def __len__(self):
        return len(self.sub_set_item_index)

    def __getitem__(self, idx):
        subset_idx = self.sub_set_index[idx]
        item_idx = self.sub_set_item_index[idx]
        scene_name = self.scene_names[idx]

        out = self.sub_sets[subset_idx][item_idx]

        org_gt_seg = out['gt_seg']
        org_gt_cls = out['gt_cls']
        org_gt_cls_dist = out['gt_cls_dist']
        org_gt_n_seg = out['gt_n_seg']
        offset = self.seg_offsets[idx]
        org_n_class = self.sub_sets[subset_idx].n_class

        gt_seg = np.zeros(shape=(org_gt_seg.shape[0],), dtype=int)  # [0, ..., n_features]
        gt_n_seg = np.zeros(shape=(self.n_class,), dtype=int)
        gt_cls = np.zeros(shape=(self.n_class,), dtype=int)
        gt_cls_dist = np.zeros(shape=(self.n_class,), dtype=float)

        # copy invalid segments
        gt_n_seg[0] = org_gt_n_seg[0]
        gt_cls[0] = org_gt_cls[0]
        gt_cls_dist[0] = org_gt_cls_dist[0]
        # print('org: ', org_n_class, org_gt_seg.shape, org_gt_n_seg.shape, org_gt_seg)

        # copy valid segments
        gt_seg[org_gt_seg > 0] = org_gt_seg[org_gt_seg > 0] + offset - 1  # [0, ..., 1023]
        gt_n_seg[offset:offset + org_n_class - 1] = org_gt_n_seg[1:]  # [0...,n_seg]
        gt_cls[offset:offset + org_n_class - 1] = org_gt_cls[1:]  # [0, ..., n_seg]
        gt_cls_dist[offset:offset + org_n_class - 1] = org_gt_cls_dist[1:]  # [0, ..., n_seg]

        out['gt_seg'] = gt_seg
        out['gt_cls'] = gt_cls
        out['gt_cls_dist'] = gt_cls_dist
        out['gt_n_seg'] = gt_n_seg

        # print('gt: ', org_n_class, gt_seg.shape, gt_n_seg.shape, gt_seg)
        out['scene_name'] = scene_name

        # out['org_gt_seg'] = org_gt_seg
        # out['org_gt_n_seg'] = org_gt_n_seg
        # out['org_gt_cls'] = org_gt_cls
        # out['org_gt_cls_dist'] = org_gt_cls_dist

        return out
