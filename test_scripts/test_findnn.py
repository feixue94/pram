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

import torch

from nets.netvald import load_netvlad


def extract_netvald_feat(image_root, image_names, model):
    feats = []
    with torch.no_grad():
        for fn in image_names:
            img = cv2.imread(osp.join(image_root, fn))
            img = img.astype(np.float32) / 255
            feat = model(torch.from_numpy(img).permute(2, 0, 1).cuda()[None])
            feats.append(feat)

    return torch.cat(feats, dim=0)  # [N, D]


if __name__ == '__main__':
    netvald = load_netvlad(
        weight_path='/scratches/flyer_2/fx221/Research/Code/third_weights/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar')
    netvald = netvald.cuda().eval()
    
    ref_image_root = '/scratches/flyer_3/fx221/dataset/Hospital/images/front'
    query_img_root = '/scratches/flyer_3/fx221/dataset/Hospital/query'

    ref_images = glob(ref_image_root + '/**/*.png', recursive=True)
    query_images = glob(query_img_root + '/**/*.png', recursive=True)

    ref_images = sorted(ref_images)
    query_feats = extract_netvald_feat(image_root=ref_image_root, image_names=ref_images, model=netvald)
