# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> loc_by_rec
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   08/02/2024 15:26
=================================================='''
import torch
from localization.multimap3d import MultiMap3D
from localization.frame import Frame
from localization.tracker import Tracker
import yaml, cv2, time
import numpy as np
import os
import os.path as osp
import threading
from recognition.vis_seg import vis_seg_point, generate_color_dic
from tools.common import resize_img
from visualization.visualizer import Visualizer
from localization.utils import read_query_info
from localization.camera import Camera


def loc_by_rec_online(rec_model, config, local_feat, img_transforms=None):
    seg_color = generate_color_dic(n_seg=2000)
    dataset_path = config['dataset_path']
    show = config['localization']['show']
    if show:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    locMap = MultiMap3D(config=config, save_dir=None)
    if config['dataset'][0] in ['Aachen']:
        viewer_config = {'scene': 'outdoor',
                         'image_size_indoor': 4,
                         'image_line_width_indoor': 8, }
    elif config['dataset'][0] in ['C']:
        viewer_config = {'scene': 'outdoor'}
    elif config['dataset'][0] in ['12Scenes', '7Scenes']:
        viewer_config = {'scene': 'indoor', }
    else:
        viewer_config = {'scene': 'outdoor',
                         'image_size_indoor': 0.4,
                         'image_line_width_indoor': 2, }

    # start tracker
    dataset_name = config['dataset'][0]
    all_scene_query_info = {}
    with open(osp.join(config['config_path'], '{:s}.yaml'.format(dataset_name)), 'r') as f:
        scene_config = yaml.load(f, Loader=yaml.Loader)

    # multiple scenes in a single dataset
    show_time = -1
    scenes = scene_config['scenes']
    for scene in scenes:
        if len(config['localization']['loc_scene_name']) > 0:
            if scene not in config['localization']['loc_scene_name']:
                continue

        query_path = osp.join(config['dataset_path'], dataset_name, scene, scene_config[scene]['query_path'])
        query_info = read_query_info(query_fn=query_path)
        all_scene_query_info[dataset_name + '/' + scene] = query_info
        image_path = osp.join(dataset_path, dataset_name, scene)
        for fn in sorted(query_info.keys()):
            # for fn in sorted(query_info.keys())[880:][::5]:  # darwinRGB-loc-outdoor-aligned
            # for fn in sorted(query_info.keys())[3161:][::5]:  # darwinRGB-loc-indoor-aligned

            # for fn in sorted(query_info.keys())[2100:][::5]: # darwinRGB-loc-outdoor
            # for fn in sorted(query_info.keys())[4360:][::5]:  # darwinRGB-loc-indoor
            # for fn in sorted(query_info.keys())[1380:]:  # Cam-Church
            # for fn in sorted(query_info.keys())[::5]: #ACUED-test2
            # for fn in sorted(query_info.keys())[2100:]:
            # for fn in sorted(query_info.keys())[4850:]:
            img = cv2.imread(osp.join(image_path, fn))  # BGR

            with torch.no_grad():
                if config['image_dim'] == 1:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_cuda = torch.from_numpy(img_gray / 255)[None].cuda().float()
                else:
                    img_cuda = torch.from_numpy(img / 255).permute(2, 0, 1).cuda().float()
                if img_transforms is not None:
                    img_cuda = img_transforms(img_cuda)[None]
                else:
                    img_cuda = img_cuda[None]

                t_start = time.time()
                encoder_out = local_feat.extract_local_global(data={'image': img_cuda},
                                                              config={'min_keypoints': 128,
                                                                      'max_keypoints': config['eval_max_keypoints'],
                                                                      }
                                                              )
                t_feat = time.time() - t_start
                t_start = time.time()
                # global_descriptors_cuda = encoder_out['global_descriptors']
                scores_cuda = encoder_out['scores'][0][None]
                kpts_cuda = encoder_out['keypoints'][0][None]
                descriptors_cuda = encoder_out['descriptors'][0][None].permute(0, 2, 1)
                _, seg_descriptors = local_feat.sample(score_map=encoder_out['score_map'],
                                                       semi_descs=encoder_out['mid_features'],
                                                       kpts=kpts_cuda[0],
                                                       norm_desc=config['norm_desc'])
                rec_out = rec_model({'scores': scores_cuda,
                                     'seg_descriptors': seg_descriptors[None].permute(0, 2, 1),
                                     'keypoints': kpts_cuda,
                                     'image': img_cuda})
                t_rec = time.time() - t_start

                pred = {
                    'scores': scores_cuda,
                    'keypoints': kpts_cuda,
                    'descriptors': descriptors_cuda,
                    # 'global_descriptors': global_descriptors_cuda,
                    'image_size': np.array([img.shape[1], img.shape[0]])[None],
                }

                camera_model, width, height, params = all_scene_query_info[dataset_name + '/' + scene][fn]
                camera = Camera(id=-1, model=camera_model, width=width, height=height, params=params)
                curr_frame = Frame(image=img, camera=camera, id=0, name=fn, scene_name=dataset_name + '/' + scene)
                curr_frame.update_features(
                    keypoints=np.hstack([pred['keypoints'][0].cpu().numpy(),
                                         pred['scores'][0].cpu().numpy().reshape(-1, 1)]),
                    descriptors=pred['descriptors'][0].cpu().numpy())

                pred = {**pred, **rec_out}
                pred_seg = torch.max(pred['prediction'], dim=2)[1]  # [B, N, C]
                pred_seg = pred_seg[0].cpu().numpy()
                kpts = kpts_cuda[0].cpu().numpy()

                img_pred_seg = vis_seg_point(img=img, kpts=kpts, segs=pred_seg, seg_color=seg_color, radius=9)
                show_text = 'kpts: {:d}'.format(kpts.shape[0])
                img_pred_seg = resize_img(img_pred_seg, nh=512)
                img_pred_seg = cv2.putText(img=img_pred_seg, text=show_text,
                                           org=(50, 30),
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1, color=(0, 0, 255),
                                           thickness=2, lineType=cv2.LINE_AA)

                if show:
                    cv2.imshow('img', img)
                    key = cv2.waitKey(show_time)
                    if key == ord('q'):
                        exit(0)
                    elif key == ord('s'):
                        show_time = -1
                    elif key == ord('c'):
                        show_time = 1

                segmentations = pred['prediction'][0]  # .cpu().numpy()  # [N, C]

                success = locMap.run(q_frame=curr_frame, q_segs=segmentations)
                if success:
                    pass
