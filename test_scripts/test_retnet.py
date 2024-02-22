# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> test_retnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   22/02/2024 15:24
=================================================='''
import numpy as np
import torch
import os
import os.path as osp
import cv2
from nets.retnet import RetNet
from nets.sfd2 import ResNet4x
import torchvision.transforms as tvf

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.Normalize(mean=RGB_mean, std=RGB_std)])


def extract_state_dict():
    weight_path = '/scratches/flyer_2/fx221/exp/glretrieve/checkpoints/model_best.pth.tar'
    weights = torch.load(weight_path, map_location='cpu')
    state_dict = weights['state_dict']
    print(weights['recalls'], weights['epoch'])

    new_state_dict = {}
    for k in state_dict.keys():
        v = state_dict[k]
        if k.find('pool') >= 0:
            new_state_dict[k[5 + 7:]] = v

    torch.save({'state_dict': new_state_dict},
               '/scratches/flyer_2/fx221/exp/glretrieve/checkpoints/model_best_retnet.pth')


def extract_features(sfd2, rnet, image):
    with torch.no_grad():
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.
        sfd2_out = sfd2.extract_local_global({'image': torch.from_numpy(image).cuda().float()[None]})
        feat = rnet(sfd2_out['mid_features'])
        return feat


if __name__ == '__main__':
    sfd2 = ResNet4x(outdim=128).cuda().eval()
    sfd2.load_state_dict(torch.load('weights/20230511_210205_resnet4x.79.pth')['state_dict'])
    rnet_weight_path = '/scratches/flyer_2/fx221/exp/glretrieve/checkpoints/model_best_retnet.pth'
    rnet = RetNet(indim=256, outdim=1024).cuda().eval()
    rnet.load_state_dict(torch.load(rnet_weight_path, map_location='cpu')['state_dict'], strict=True)

    root_dir = '/scratches/flyer_3/fx221/dataset/ACUED/cued'
    query_img_dir = osp.join(root_dir, 'query')
    query_fns = sorted(os.listdir(query_img_dir))
    db_img_dir = osp.join(root_dir, 'image')

    vrf_data = np.load(
        '/scratches/flyer_3/fx221/exp/localizer/resnet4x-20230511-210205-pho-0005-gm/ACUED/cued/point3D_vrf_n128_xy_birch.npy',
        allow_pickle=True)[()]
    db_img_list = [vrf_data[k][0]['image_name'] for k in vrf_data.keys()]
    db_images = [cv2.imread(osp.join(root_dir, fn)) for fn in db_img_list]
    db_feats = [extract_features(sfd2, rnet, img) for img in db_images]
    db_feats = torch.cat(db_feats, dim=0)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    for qfn in query_fns:
        qimg = cv2.imread(osp.join(query_img_dir, qfn))
        with torch.no_grad():
            qfeat = extract_features(sfd2, rnet, qimg)
            dist = torch.einsum('id,jd->ij', qfeat, db_feats)
            kdists, kinds = torch.topk(dist, k=5, dim=1, largest=True)
            kinds = kinds.cpu().numpy()[0]
            for i in kinds:
                cv2.imshow('img', np.hstack([qimg, db_images[i]]))
                cv2.waitKey(0)
