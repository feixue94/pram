# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> test_script
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   19/02/2024 14:36
=================================================='''
import os
import cv2
import numpy as np
import torch
from copy import deepcopy
from localization.extract_features import confs as feat_confs
from localization.extract_features import get_model
from localization.match_features_batch import confs as matcher_confs
from localization.base_model import dynamic_load
import localization.matchers as matchers
from recognition.vis_seg import plot_matches, plot_kpts

if __name__ == '__main__':
    conf = feat_confs['sfd2']
    sfd2, extractor = get_model(model_name=conf['model']['name'], weight_path=conf["model"]["model_fn"],
                                outdim=conf['model']['outdim'])
    sfd2 = sfd2.cuda().eval()
    print("model: ", sfd2)

    imp = dynamic_load(matchers, 'gml')
    imp = imp(matcher_confs['gml']['model']).eval().cuda()

    image_dir = '/scratches/flyer_3/fx221/exp/hospital/2024-01-19-17-14-09_3f_short'

    all_fns = os.listdir(image_dir)
    all_fns = sorted(all_fns)
    last_kpts = None
    last_descs = None
    last_scores = None
    last_image = None
    last_image_name = None

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    for i in range(len(all_fns)):
        # if i % 1 > 0:
        #     continue
        fn = all_fns[i]
        if fn.find('png') < 0:
            continue

        img = cv2.imread(os.path.join(image_dir, fn))
        with torch.no_grad():
            feats_out = extractor(sfd2,
                                  img=torch.from_numpy(img.transpose((2, 0, 1)) / 255.).float(),
                                  topK=conf["model"]["max_keypoints"],
                                  mask=None,
                                  conf_th=0.01,
                                  scales=conf["model"]["scales"],
                                  )

        kpts = feats_out['keypoints']
        descs = feats_out['descriptors']

        if last_kpts is not None:
            matches = imp(data={
                'keypoints0': torch.from_numpy(last_kpts[None]).float().cuda(),
                'descriptors0': torch.from_numpy(last_descs[None]).float().cuda(),
                'image_shape0': (1, 3, last_image.shape[0], last_image.shape[1]),

                'keypoints1': torch.from_numpy(kpts[None]).float().cuda(),
                'descriptors1': torch.from_numpy(descs[None]).float().cuda(),
                'image_shape1': (1, 3, img.shape[0], img.shape[1]),
            })['matches0'][0].cpu().numpy()
            print('matches: ', matches.shape)

            valid = (matches >= 0)

            last_kpt_img = plot_kpts(img=last_image, kpts=last_kpts)
            last_kpt_img = cv2.putText(last_kpt_img, '{:s},kpts:{:d}'.format(last_image_name.split('.')[0],
                                                                             last_kpts.shape[0]),
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 0, 255), 2
                                       )
            kpt_img = plot_kpts(img=img, kpts=kpts)
            kpt_img = cv2.putText(kpt_img, '{:s},kpts:{:d}'.format(fn.split('.')[0],
                                                                   kpts.shape[0]),
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 0, 255), 2
                                  )

            match_img = plot_matches(img1=last_image,
                                     pts1=last_kpts[valid],
                                     img2=img, pts2=kpts[matches[valid]],
                                     inliers=np.array([True for v in range(np.sum(valid))]))

            match_img = cv2.putText(match_img, '#matches:{:d}'.format(np.sum(valid)),
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (0, 0, 255), 3)

            show_img = np.hstack([last_kpt_img, kpt_img])
            show_img = np.vstack([show_img, match_img])

            cv2.imwrite(os.path.join('/scratches/flyer_3/fx221/exp/hospital/match_image', fn), show_img)
            cv2.imshow('image', show_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
        if last_image is None or i % 30 == 0:
            last_image = deepcopy(img)
            last_kpts = kpts
            last_descs = descs
            last_image_name = fn
