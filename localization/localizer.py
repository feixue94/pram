# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> hloc
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 16:45
=================================================='''

import os
import os.path as osp
from tqdm import tqdm
import argparse
import time
import logging
import h5py
import numpy as np
from pathlib import Path
from colmap_utils.read_write_model import read_model
from colmap_utils.parsers import parse_image_lists_with_intrinsics
# localization
from localization.match_features import confs
from localization.base_model import dynamic_load
from localization import matchers
from localization.utils import compute_pose_error, read_gt_pose, read_retrieval_results
from localization.pose_estimator import pose_estimator_hloc, pose_estimator_iterative


def run(args):
    if args.gt_pose_fn is not None:
        gt_poses = read_gt_pose(path=args.gt_pose_fn)
    else:
        gt_poses = {}
    retrievals = read_retrieval_results(args.retrieval)

    save_root = args.save_root  # path to save
    os.makedirs(save_root, exist_ok=True)
    matcher_name = args.matcher_method  # matching method
    print('matcher: ', confs[args.matcher_method]['model']['name'])
    Model = dynamic_load(matchers, confs[args.matcher_method]['model']['name'])
    matcher = Model(confs[args.matcher_method]['model']).eval().cuda()

    local_feat_name = args.features.as_posix().split("/")[-1].split(".")[0]  # name of local features
    save_fn = '{:s}_{:s}'.format(local_feat_name, matcher_name)
    if args.use_hloc:
        save_fn = 'hloc_' + save_fn
    save_fn = osp.join(save_root, save_fn)

    queries = parse_image_lists_with_intrinsics(args.queries)
    _, db_images, points3D = read_model(str(args.reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    feature_file = h5py.File(args.features, 'r')

    tag = ''
    if args.do_covisible_opt:
        tag = tag + "_o" + str(int(args.obs_thresh)) + 'op' + str(int(args.covisibility_frame))
        tag = tag + "th" + str(int(args.opt_thresh)) + "r" + str(args.radius)
        if args.iters > 0:
            tag = tag + "i" + str(int(args.iters))

    log_fn = save_fn + tag
    vis_dir = save_fn + tag
    results = save_fn + tag

    full_log_fn = log_fn + '_full.log'
    loc_log_fn = log_fn + '_loc.npy'
    results = Path(results + '.txt')
    vis_dir = Path(vis_dir)
    if vis_dir is not None:
        Path(vis_dir).mkdir(exist_ok=True)
    print("save_fn: ", log_fn)

    logging.info('Starting localization...')
    poses = {}
    failed_cases = []
    n_total = 0
    n_failed = 0
    full_log_info = ''
    loc_results = {}

    error_ths = ((0.25, 2), (0.5, 5), (5, 10))
    success = [0, 0, 0]
    total_loc_time = []

    for qname, qinfo in tqdm(queries):
        kpq = feature_file[qname]['keypoints'].__array__()
        n_total += 1
        time_start = time.time()

        if qname in retrievals.keys():
            cans = retrievals[qname]
            db_ids = [db_name_to_id[v] for v in cans]
        else:
            cans = []
            db_ids = []
        time_coarse = time.time()

        if args.use_hloc:
            output = pose_estimator_hloc(qname=qname, qinfo=qinfo, db_ids=db_ids, db_images=db_images,
                                         points3D=points3D,
                                         feature_file=feature_file,
                                         thresh=args.ransac_thresh,
                                         image_dir=args.image_dir,
                                         matcher=matcher,
                                         log_info='',
                                         query_img_prefix='',
                                         db_img_prefix='')
        else:  # should be faster and more accurate than hloc
            output = pose_estimator_iterative(qname=qname,
                                              qinfo=qinfo,
                                              matcher=matcher,
                                              db_ids=db_ids,
                                              db_images=db_images,
                                              points3D=points3D,
                                              feature_file=feature_file,
                                              thresh=args.ransac_thresh,
                                              image_dir=args.image_dir,
                                              do_covisibility_opt=args.do_covisible_opt,
                                              covisibility_frame=args.covisibility_frame,
                                              log_info='',
                                              inlier_th=args.inlier_thresh,
                                              obs_th=args.obs_thresh,
                                              opt_th=args.opt_thresh,
                                              gt_qvec=gt_poses[qname]['qvec'] if qname in gt_poses.keys() else None,
                                              gt_tvec=gt_poses[qname]['tvec'] if qname in gt_poses.keys() else None,
                                              query_img_prefix='query',
                                              db_img_prefix='database',
                                              )
        time_full = time.time()

        qvec = output['qvec']
        tvec = output['tvec']
        loc_time = output['time']
        total_loc_time.append(loc_time)

        poses[qname] = (qvec, tvec)
        print_text = "All {:d}/{:d} failed cases, time[cs/fn]: {:.2f}/{:.2f}".format(
            n_failed, n_total,
            time_coarse - time_start,
            time_full - time_coarse,
        )

        if qname in gt_poses.keys():
            gt_qvec = gt_poses[qname]['qvec']
            gt_tvec = gt_poses[qname]['tvec']

            q_error, t_error, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=gt_qvec, gt_tcw=gt_tvec)

            for error_idx, th in enumerate(error_ths):
                if t_error <= th[0] and q_error <= th[1]:
                    success[error_idx] += 1
            print_text += (
                ', q_error:{:.2f} t_error:{:.2f} {:d}/{:d}/{:d}/{:d}, time: {:.2f}, {:d}pts'.format(q_error, t_error,
                                                                                                    success[0],
                                                                                                    success[1],
                                                                                                    success[2], n_total,
                                                                                                    loc_time,
                                                                                                    kpq.shape[0]))
        if output['num_inliers'] == 0:
            failed_cases.append(qname)

        loc_results[qname] = {
            'keypoints_query': output['keypoints_query'],
            'points3D_ids': output['points3D_ids'],
        }
        full_log_info = full_log_info + output['log_info']
        full_log_info += (print_text + "\n")
        print(print_text)

    logs_path = f'{results}.failed'
    with open(logs_path, 'w') as f:
        for v in failed_cases:
            print(v)
            f.write(v + "\n")

    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    # logging.info(f'Mean loc time: {np.mean(total_loc_time)}...')
    print('Mean loc time: {:.2f}...'.format(np.mean(total_loc_time)))
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q
            f.write(f'{name} {qvec} {tvec}\n')

    with open(full_log_fn, 'w') as f:
        f.write(full_log_info)

    np.save(loc_log_fn, loc_results)
    print('Save logs to ', loc_log_fn)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12)
    parser.add_argument('--covisibility_frame', type=int, default=50)
    parser.add_argument('--do_covisible_opt', action='store_true')
    parser.add_argument('--use_hloc', action='store_true')
    parser.add_argument('--matcher_method', type=str, default="NNM")
    parser.add_argument('--inlier_thresh', type=int, default=50)
    parser.add_argument('--obs_thresh', type=float, default=3)
    parser.add_argument('--opt_thresh', type=float, default=12)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--retrieval', type=Path, default=None)
    parser.add_argument('--gt_pose_fn', type=str, default=None)

    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    run(args=args)
