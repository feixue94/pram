# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram-dev -> test_findnn
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   09/12/2024 11:46
=================================================='''
import cv2
import os
import os.path as osp
import numpy as np
from glob import glob
from tqdm import tqdm

import torch

from nets.netvald import load_netvlad, extract_netvlad_feat


def extract_netvald_feats(image_paths, model):
    feats = []
    with torch.no_grad():
        for fn in tqdm(image_paths, total=len(image_paths)):
            img = cv2.imread(fn)
            img = img.astype(np.float32) / 255
            feat = extract_netvlad_feat(torch.from_numpy(img).permute(2, 0, 1).cuda()[None], model=model)
            # print('feat: ', feat.shape)
            feats.append(feat)

    return torch.cat(feats, dim=0)  # [N, D]


if __name__ == '__main__':
    netvald = load_netvlad(
        weight_path='/scratches/flyer_2/fx221/Research/Code/third_weights/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar')
    netvald = netvald.cuda().eval()

    ref_image_root = '/scratches/flyer_3/fx221/dataset/Hospital/images/front'
    query_img_root = '/scratches/flyer_3/fx221/dataset/Hospital/query'

    ref_images = glob(ref_image_root + '/**/*.png', recursive=True)
    print('Find {} reference images in {}'.format(len(ref_images), ref_image_root))
    query_images = glob(query_img_root + '/**/*.png', recursive=True)
    print('Find {} query images in {}'.format(len(query_images), query_img_root))
    # print(ref_images)

    ref_images = sorted(ref_images)
    ref_feats = extract_netvald_feats(image_paths=ref_images, model=netvald)  # [N, D]
    query_feats = extract_netvald_feats(image_paths=query_images, model=netvald)  # [M, D]
    pairs = {}
    dist = query_feats @ ref_feats.t()

    topk = 20
    values, ids = torch.topk(dist, dim=1, k=topk, largest=True)
    print('dist: ', dist.shape)
    # print('values: ', values)
    # print('ids: ', ids)
    for i in range(len(query_images)):
        q_name = osp.relpath(query_images[i], query_img_root)
        # q_img = cv2.imread(query_images[i])
        candidates = []
        for k in range(topk):
            candidates.append(osp.relpath(ref_images[ids[i][k]], ref_image_root))
            c_img = cv2.imread(ref_images[ids[i][k]])
            # img = np.hstack([q_img, c_img])
            #
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        pairs[q_name] = candidates

    with open('/scratches/flyer_3/fx221/dataset/Hospital/query/query-pairs-bag1-front-netvald-20.txt', 'w') as f:
        for k in pairs.keys():
            for c in pairs[k]:
                f.write(k + ' ' + c + '\n')
