# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> loc_by_rec
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   08/02/2024 15:26
=================================================='''
import torch
from torch.autograd import Variable
from localization.multimap3d import MultiMap3D
from localization.frame import Frame
import yaml, cv2, time
import numpy as np
import os.path as osp
import threading
import os
from tqdm import tqdm
from recognition.vis_seg import vis_seg_point, generate_color_dic
from tools.metrics import compute_iou, compute_precision
from localization.viewer import Viewer
from localization.tracker import Tracker
from localization.utils import read_query_info
from localization.camera import Camera


def loc_by_rec_eval(rec_model, loader, config, local_feat, img_transforms=None):
    n_epoch = int(config['weight_path'].split('.')[1])
    save_fn = osp.join(config['localization']['save_path'],
                       config['weight_path'].split('/')[0] + '_{:d}'.format(n_epoch) + '_{:d}'.format(
                           config['feat_dim']))
    tag = 'k{:d}_th{:d}_mm{:d}_mi{:d}'.format(config['localization']['seg_k'], config['localization']['threshold'],
                                              config['localization']['min_matches'],
                                              config['localization']['min_inliers'])
    if config['localization']['do_refinement']:
        tag += '_op{:d}'.format(config['localization']['covisibility_frame'])
    if config['localization']['with_compress']:
        tag += '_comp'

    save_fn = save_fn + '_' + tag

    save = config['localization']['save']
    save = config['localization']['save']
    if save:
        save_dir = save_fn
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    seg_color = generate_color_dic(n_seg=2000)
    dataset_path = config['dataset_path']
    show = config['localization']['show']
    if show:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    locMap = MultiMap3D(config=config, save_dir=None)
    # start tracker
    mTracker = Tracker(locMap=locMap, matcher=locMap.matcher, config=config)

    dataset_name = config['dataset'][0]
    all_scene_query_info = {}
    with open(osp.join(config['config_path'], '{:s}.yaml'.format(dataset_name)), 'r') as f:
        scene_config = yaml.load(f, Loader=yaml.Loader)
    scenes = scene_config['scenes']
    for scene in scenes:
        query_path = osp.join(config['dataset_path'], dataset_name, scene, scene_config[scene]['query_path'])
        query_info = read_query_info(query_fn=query_path)
        all_scene_query_info[dataset_name + '/' + scene] = query_info
        # print(scene, query_info.keys())

    tracking = False

    full_log = ''
    failed_cases = []
    success_cases = []
    poses = {}
    err_ths_cnt = [0, 0, 0, 0]

    seg_results = {}
    time_results = {
        'feat': [],
        'rec': [],
        'loc': [],
        'ref': [],
        'total': [],
    }
    n_total = 0

    loc_scene_names = config['localization']['loc_scene_name']
    for bid, pred in tqdm(enumerate(loader), total=len(loader)):
        pred = loader[bid]
        with torch.no_grad():
            for k in pred:
                if k.find('name') >= 0:
                    continue
                if k != 'image0' and k != 'image1' and k != 'depth0' and k != 'depth1':
                    if type(pred[k]) == np.ndarray:
                        pred[k] = Variable(torch.from_numpy(pred[k]).float().cuda())[None]
                    elif type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].float().cuda())
                    elif type(pred[k]) == list:
                        continue
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).float().cuda())
            image_name = pred['file_name']  # [0]
            scene_name = pred['scene_name']  # [0]  # dataset_scene
            if len(loc_scene_names) > 0:
                skip = True
                for loc_scene in loc_scene_names:
                    if scene_name.find(loc_scene) > 0:
                        skip = False
                        break
                if skip:
                    continue
            print('scene: ', scene_name, image_name)

            n_total += 1
            with torch.no_grad():
                img = pred['image']
                while isinstance(img, list):
                    img = img[0]

                new_im = torch.from_numpy(img).permute(2, 0, 1).cuda().float()
                if img_transforms is not None:
                    new_im = img_transforms(new_im)[None]
                else:
                    new_im = new_im[None]
                img = (img * 255).astype(np.uint8)

                fn = image_name
                camera_model, width, height, params = all_scene_query_info[scene_name][fn]
                camera = Camera(id=-1, model=camera_model, width=width, height=height, params=params)
                curr_frame = Frame(image=img, camera=camera, id=0, name=fn, scene_name=scene_name)
                gt_sub_map = locMap.sub_maps[curr_frame.scene_name]
                if gt_sub_map.gt_poses is not None and curr_frame.name in gt_sub_map.gt_poses.keys():
                    curr_frame.gt_qvec = gt_sub_map.gt_poses[curr_frame.name]['qvec']
                    curr_frame.gt_tvec = gt_sub_map.gt_poses[curr_frame.name]['tvec']

                    t_start = time.time()
                    encoder_out = local_feat.extract_local_global(data={'image': new_im},
                                                                  config=
                                                                  {
                                                                      # 'min_keypoints': 128,
                                                                      'max_keypoints': config['eval_max_keypoints'],
                                                                  }
                                                                  )
                    t_feat = time.time() - t_start
                    # global_descriptors_cuda = encoder_out['global_descriptors']
                    # scores_cuda = encoder_out['scores'][0][None]
                    # kpts_cuda = encoder_out['keypoints'][0][None]
                    # descriptors_cuda = encoder_out['descriptors'][0][None].permute(0, 2, 1)

                    sparse_scores = pred['scores']
                    sparse_descs = pred['descriptors']
                    sparse_kpts = pred['keypoints']
                    gt_seg = pred['gt_seg']

                    curr_frame.add_keypoints(keypoints=np.hstack([sparse_kpts[0].cpu().numpy(),
                                                                  sparse_scores[0].cpu().numpy().reshape(-1, 1)]),
                                             descriptors=sparse_descs[0].cpu().numpy())
                    curr_frame.time_feat = t_feat

                    t_start = time.time()
                    _, seg_descriptors = local_feat.sample(score_map=encoder_out['score_map'],
                                                           semi_descs=encoder_out['mid_features'],
                                                           # kpts=kpts_cuda[0],
                                                           kpts=sparse_kpts[0],
                                                           norm_desc=config['norm_desc'])
                    rec_out = rec_model({'scores': sparse_scores,
                                         'seg_descriptors': seg_descriptors[None].permute(0, 2, 1),
                                         'keypoints': sparse_kpts,
                                         'image': new_im})
                    t_rec = time.time() - t_start
                    curr_frame.time_rec = t_rec

                    pred = {
                        # 'scores': scores_cuda,
                        # 'keypoints': kpts_cuda,
                        # 'descriptors': descriptors_cuda,
                        # 'global_descriptors': global_descriptors_cuda,
                        'image_size': np.array([img.shape[1], img.shape[0]])[None],
                    }

                    pred = {**pred, **rec_out}
                    pred_seg = torch.max(pred['prediction'], dim=2)[1]  # [B, N, C]

                    pred_seg = pred_seg[0].cpu().numpy()
                    kpts = sparse_kpts[0].cpu().numpy()
                    img_pred_seg = vis_seg_point(img=img, kpts=kpts, segs=pred_seg, seg_color=seg_color, radius=9)
                    show_text = 'kpts: {:d}'.format(kpts.shape[0])
                    img_pred_seg = cv2.putText(img=img_pred_seg, text=show_text,
                                               org=(50, 30),
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1, color=(0, 0, 255),
                                               thickness=2, lineType=cv2.LINE_AA)
                    curr_frame.image_rec = img_pred_seg

                    if show:
                        cv2.imshow('img', img)
                        key = cv2.waitKey(1)
                        if key == ord('q'):
                            exit(0)
                        elif key == ord('s'):
                            show_time = -1
                        elif key == ord('c'):
                            show_time = 1

                    segmentations = pred['prediction'][0]  # .cpu().numpy()  # [N, C]
                    curr_frame.add_segmentations(segmentations=segmentations,
                                                 filtering_threshold=config['localization']['pre_filtering_th'])

                    # Step1: do tracker first
                    success = not mTracker.lost and tracking
                    if success:
                        success = mTracker.run(frame=curr_frame)
                    if not success:
                        success = locMap.run(q_frame=curr_frame)
                    if success:
                        curr_frame.update_point3ds()
                        if tracking:
                            mTracker.lost = False
                            mTracker.last_frame = curr_frame
                    # '''
                    pred_seg = torch.max(pred['prediction'], dim=-1)[1]  # [B, N, C]
                    pred_seg = pred_seg[0].cpu().numpy()
                    gt_seg = gt_seg[0].cpu().numpy()
                    iou = compute_iou(pred=pred_seg, target=gt_seg, n_class=pred_seg.shape[0],
                                      ignored_ids=[0])  # 0 - background
                    prec = compute_precision(pred=pred_seg, target=gt_seg, ignored_ids=[0])

                    kpts = sparse_kpts[0].cpu().numpy()
                    if scene not in seg_results.keys():
                        seg_results[scene] = {
                            'day': {
                                'prec': [],
                                'iou': [],
                                'kpts': [],
                            },
                            'night': {
                                'prec': [],
                                'iou': [],
                                'kpts': [],

                            }
                        }
                    if fn.find('night') >= 0:
                        seg_results[scene]['night']['prec'].append(prec)
                        seg_results[scene]['night']['iou'].append(iou)
                        seg_results[scene]['night']['kpts'].append(kpts.shape[0])
                    else:
                        seg_results[scene]['day']['prec'].append(prec)
                        seg_results[scene]['day']['iou'].append(iou)
                        seg_results[scene]['day']['kpts'].append(kpts.shape[0])

                    print_text = 'name: {:s}, kpts: {:d}, iou: {:.3f}, prec: {:.3f}'.format(fn, kpts.shape[0], iou,
                                                                                            prec)
                    print(print_text)
                    # '''

                    t_feat = curr_frame.time_feat
                    t_rec = curr_frame.time_rec
                    t_loc = curr_frame.time_loc
                    t_ref = curr_frame.time_ref
                    t_total = t_feat + t_rec + t_loc + t_ref
                    time_results['feat'].append(t_feat)
                    time_results['rec'].append(t_rec)
                    time_results['loc'].append(t_loc)
                    time_results['ref'].append(t_ref)
                    time_results['total'].append(t_total)

                    poses[scene + '/' + fn] = (curr_frame.qvec, curr_frame.tvec)
                    q_err, t_err = curr_frame.compute_pose_error()
                    if q_err <= 5 and t_err <= 0.05:
                        err_ths_cnt[0] = err_ths_cnt[0] + 1
                    if q_err <= 2 and t_err <= 0.25:
                        err_ths_cnt[1] = err_ths_cnt[1] + 1
                    if q_err <= 5 and t_err <= 0.5:
                        err_ths_cnt[2] = err_ths_cnt[2] + 1
                    if q_err <= 10 and t_err <= 5:
                        err_ths_cnt[3] = err_ths_cnt[3] + 1

                    if success:
                        success_cases.append(scene + '/' + fn)
                        print_text = 'qname: {:s} localization success {:d}/{:d}, q_err: {:.2f}, t_err: {:.2f}, {:d}/{:d}/{:d}/{:d}/{:d}, time: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(
                            scene + '/' + fn, len(success_cases), n_total, q_err, t_err, err_ths_cnt[0],
                            err_ths_cnt[1],
                            err_ths_cnt[2],
                            err_ths_cnt[3],
                            n_total,
                            t_feat, t_rec, t_loc, t_ref, t_total
                        )
                    else:
                        failed_cases.append(scene + '/' + fn)
                        print_text = 'qname: {:s} localization fail {:d}/{:d}, q_err: {:.2f}, t_err: {:.2f}, {:d}/{:d}/{:d}/{:d}/{:d}, time: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(
                            scene + '/' + fn, len(failed_cases), n_total, q_err, t_err, err_ths_cnt[0],
                            err_ths_cnt[1],
                            err_ths_cnt[2],
                            err_ths_cnt[3],
                            n_total, t_feat, t_rec, t_loc, t_ref, t_total)
                    print(print_text)
