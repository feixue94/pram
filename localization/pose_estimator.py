# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> pose_estimation
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   08/02/2024 11:01
=================================================='''
import torch
import numpy as np
import pycolmap
import cv2
import os
import time
import os.path as osp
from collections import defaultdict


def get_covisibility_frames(frame_id, all_images, points3D, covisibility_frame=50):
    observed = all_images[frame_id].point3D_ids
    covis = defaultdict(int)
    for pid in observed:
        if pid == -1:
            continue
        for img_id in points3D[pid].image_ids:
            if img_id != frame_id:
                covis[img_id] += 1

    print('Find {:d} connected frames'.format(len(covis.keys())))

    covis_ids = np.array(list(covis.keys()))
    covis_num = np.array([covis[i] for i in covis_ids])

    if len(covis_ids) <= covisibility_frame:
        sel_covis_ids = covis_ids[np.argsort(-covis_num)]
    else:
        ind_top = np.argpartition(covis_num, -covisibility_frame)
        ind_top = ind_top[-covisibility_frame:]  # unsorted top k
        ind_top = ind_top[np.argsort(-covis_num[ind_top])]
        sel_covis_ids = [covis_ids[i] for i in ind_top]

    print('Retain {:d} valid connected frames'.format(len(sel_covis_ids)))
    return sel_covis_ids


def feature_matching(query_data, db_data, matcher):
    db_3D_ids = db_data['db_3D_ids']
    if db_3D_ids is None:
        with torch.no_grad():
            match_data = {
                'keypoints0': torch.from_numpy(query_data['keypoints'])[None].float().cuda(),
                'scores0': torch.from_numpy(query_data['scores'])[None].float().cuda(),
                'descriptors0': torch.from_numpy(query_data['descriptors'])[None].float().cuda(),
                'image0': torch.empty((1, 1,) + tuple(query_data['image_size'])[::-1]),

                'keypoints1': torch.from_numpy(db_data['keypoints'])[None].float().cuda(),
                'scores1': torch.from_numpy(db_data['scores'])[None].float().cuda(),
                'descriptors1': torch.from_numpy(db_data['descriptors'])[None].float().cuda(),  # [B, N, D]
                'image1': torch.empty((1, 1,) + tuple(db_data['image_size'])[::-1]),
            }
            matches = matcher(match_data)['matches0'][0].cpu().numpy()
            del match_data
    else:
        masks = (db_3D_ids != -1)
        valid_ids = [i for i in range(masks.shape[0]) if masks[i]]
        if len(valid_ids) == 0:
            return np.zeros(shape=(query_data['keypoints'].shape[0],), dtype=int) - 1
        with torch.no_grad():
            match_data = {
                'keypoints0': torch.from_numpy(query_data['keypoints'])[None].float().cuda(),
                'scores0': torch.from_numpy(query_data['scores'])[None].float().cuda(),
                'descriptors0': torch.from_numpy(query_data['descriptors'])[None].float().cuda(),
                'image0': torch.empty((1, 1,) + tuple(query_data['image_size'])[::-1]),

                'keypoints1': torch.from_numpy(db_data['keypoints'])[masks][None].float().cuda(),
                'scores1': torch.from_numpy(db_data['scores'])[masks][None].float().cuda(),
                'descriptors1': torch.from_numpy(db_data['descriptors'][masks])[None].float().cuda(),
                'image1': torch.empty((1, 1,) + tuple(db_data['image_size'])[::-1]),
            }
            matches = matcher(match_data)['matches0'][0].cpu().numpy()
            del match_data

        for i in range(matches.shape[0]):
            if matches[i] >= 0:
                matches[i] = valid_ids[matches[i]]

    return matches


def find_2D_3D_matches(query_data, db_id, points3D, feature_file, db_images, matcher, obs_th=0):
    kpq = query_data['keypoints']
    db_name = db_images[db_id].name
    kpdb = feature_file[db_name]['keypoints'][()]
    desc_db = feature_file[db_name]["descriptors"][()]
    desc_db = desc_db.transpose()

    # print('db_desc: ', desc_db.shape, query_data['descriptors'].shape)

    points3D_ids = db_images[db_id].point3D_ids
    matches = feature_matching(query_data=query_data,
                               db_data={
                                   'keypoints': kpdb,
                                   'scores': feature_file[db_name]['scores'][()],
                                   'descriptors': desc_db,
                                   'db_3D_ids': points3D_ids,
                                   'image_size': feature_file[db_name]['image_size'][()]
                               },
                               matcher=matcher)
    mkpdb = []
    mp3d_ids = []
    q_ids = []
    mkpq = []
    mp3d = []
    valid_matches = []
    for idx in range(matches.shape[0]):
        if matches[idx] == -1:
            continue
        if points3D_ids[matches[idx]] == -1:
            continue
        id_3D = points3D_ids[matches[idx]]

        # reject 3d points without enough observations
        if len(points3D[id_3D].image_ids) < obs_th:
            continue
        mp3d.append(points3D[id_3D].xyz)
        mp3d_ids.append(id_3D)

        mkpq.append(kpq[idx])
        mkpdb.append(kpdb[matches[idx]])
        q_ids.append(idx)
        valid_matches.append(matches[idx])

    mp3d = np.array(mp3d, float).reshape(-1, 3)
    mkpq = np.array(mkpq, float).reshape(-1, 2) + 0.5
    return mp3d, mkpq, mp3d_ids, q_ids


# hfnet, cvpr 2019
def pose_estimator_hloc(qname, qinfo, db_ids, db_images, points3D,
                        feature_file,
                        thresh,
                        image_dir,
                        matcher,
                        log_info=None,
                        query_img_prefix='',
                        db_img_prefix=''):
    kpq = feature_file[qname]['keypoints'][()]
    score_q = feature_file[qname]['scores'][()]
    desc_q = feature_file[qname]['descriptors'][()]
    desc_q = desc_q.transpose()
    imgsize_q = feature_file[qname]['image_size'][()]
    query_data = {
        'keypoints': kpq,
        'scores': score_q,
        'descriptors': desc_q,
        'image_size': imgsize_q,
    }

    camera_model, width, height, params = qinfo
    cam = pycolmap.Camera(model=camera_model, width=width, height=height, params=params)
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }
    all_mkpts = []
    all_mp3ds = []
    all_points3D_ids = []
    best_db_id = db_ids[0]
    best_db_name = db_images[best_db_id].name

    t_start = time.time()

    for cluster_idx, db_id in enumerate(db_ids):
        mp3d, mkpq, mp3d_ids, q_ids = find_2D_3D_matches(
            query_data=query_data,
            db_id=db_id,
            points3D=points3D,
            feature_file=feature_file,
            db_images=db_images,
            matcher=matcher,
            obs_th=3)
        if mp3d.shape[0] > 0:
            all_mkpts.append(mkpq)
            all_mp3ds.append(mp3d)
            all_points3D_ids = all_points3D_ids + mp3d_ids

    if len(all_mkpts) == 0:
        print_text = 'Localize {:s} failed, but use the pose of {:s} as approximation'.format(qname, best_db_name)
        print(print_text)
        if log_info is not None:
            log_info = log_info + print_text + '\n'

        qvec = db_images[best_db_id].qvec
        tvec = db_images[best_db_id].tvec

        return {
            'qvec': qvec,
            'tvec': tvec,
            'log_info': log_info,
            'qname': qname,
            'dbname': best_db_name,
            'num_inliers': 0,
            'order': -1,
            'keypoints_query': np.array([]),
            'points3D_ids': [],
            'time': time.time() - t_start,
        }

    all_mkpts = np.vstack(all_mkpts)
    all_mp3ds = np.vstack(all_mp3ds)

    ret = pycolmap.absolute_pose_estimation(all_mkpts, all_mp3ds, cam,
                                            estimation_options={
                                                "ransac": {"max_error": thresh}},
                                            refinement_options={},
                                            )
    if ret is None:
        ret = {'success': False, }
    else:
        ret['success'] = True
        ret['qvec'] = ret['cam_from_world'].rotation.quat[[3, 0, 1, 2]]
        ret['tvec'] = ret['cam_from_world'].translation
    success = ret['success']

    if success:
        print_text = 'qname: {:s} localization success with {:d}/{:d} inliers'.format(qname, ret['num_inliers'],
                                                                                      all_mp3ds.shape[0])
        print(print_text)
        if log_info is not None:
            log_info = log_info + print_text + '\n'

        qvec = ret['qvec']
        tvec = ret['tvec']
        ret['cfg'] = cfg
        num_inliers = ret['num_inliers']
        inliers = ret['inliers']
        return {
            'qvec': qvec,
            'tvec': tvec,
            'log_info': log_info,
            'qname': qname,
            'dbname': best_db_name,
            'num_inliers': num_inliers,
            'order': -1,
            'keypoints_query': np.array([all_mkpts[i] for i in range(len(inliers)) if inliers[i]]),
            'points3D_ids': [all_points3D_ids[i] for i in range(len(inliers)) if inliers[i]],
            'time': time.time() - t_start,
        }
    else:
        print_text = 'Localize {:s} failed, but use the pose of {:s} as approximation'.format(qname, best_db_name)
        print(print_text)
        if log_info is not None:
            log_info = log_info + print_text + '\n'

        qvec = db_images[best_db_id].qvec
        tvec = db_images[best_db_id].tvec

        return {
            'qvec': qvec,
            'tvec': tvec,
            'log_info': log_info,
            'qname': qname,
            'dbname': best_db_name,
            'num_inliers': 0,
            'order': -1,
            'keypoints_query': np.array([]),
            'points3D_ids': [],
            'time': time.time() - t_start,
        }


def pose_refinement(query_data,
                    query_cam, feature_file, db_frame_id, db_images, points3D, matcher,
                    covisibility_frame=50,
                    obs_th=3,
                    opt_th=12,
                    qvec=None,
                    tvec=None,
                    log_info='',
                    **kwargs,
                    ):
    db_ids = get_covisibility_frames(frame_id=db_frame_id, all_images=db_images, points3D=points3D,
                                     covisibility_frame=covisibility_frame)

    mp3d = []
    mkpq = []
    mkpdb = []
    all_3D_ids = []
    all_score_q = []
    kpq = query_data['keypoints']
    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        kpdb = feature_file[db_name]['keypoints'][()]
        scores_db = feature_file[db_name]['scores'][()]
        imgsize_db = feature_file[db_name]['image_size'][()]
        desc_db = feature_file[db_name]["descriptors"][()]
        desc_db = desc_db.transpose()

        points3D_ids = db_images[db_id].point3D_ids
        if points3D_ids.size == 0:
            print("No 3D points in this db image: ", db_name)
            continue

        matches = feature_matching(query_data=query_data,
                                   db_data={'keypoints': kpdb,
                                            'scores': scores_db,
                                            'descriptors': desc_db,
                                            'image_size': imgsize_db,
                                            'db_3D_ids': points3D_ids,
                                            },
                                   matcher=matcher,
                                   )
        valid = np.where(matches > -1)[0]
        valid = valid[points3D_ids[matches[valid]] != -1]
        inliers = []
        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            if len(points3D[id_3D].image_ids) < obs_th:
                continue

            inliers.append(True)

            mp3d.append(points3D[id_3D].xyz)
            mkpq.append(kpq[idx])
            mkpdb.append(kpdb[matches[idx]])
            all_3D_ids.append(id_3D)

    mp3d = np.array(mp3d, float).reshape(-1, 3)
    mkpq = np.array(mkpq, float).reshape(-1, 2) + 0.5
    print_text = 'Get {:d} covisible frames with {:d} matches from cluster optimization'.format(len(db_ids),
                                                                                                mp3d.shape[0])
    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')

    # cam = pycolmap.Camera(model=cfg['model'], params=cfg['params'])
    ret = pycolmap.absolute_pose_estimation(mkpq, mp3d,
                                            query_cam,
                                            estimation_options={
                                                "ransac": {"max_error": opt_th}},
                                            refinement_options={},
                                            )
    if ret is None:
        ret = {'success': False, }
    else:
        ret['success'] = True
        ret['qvec'] = ret['cam_from_world'].rotation.quat[[3, 0, 1, 2]]
        ret['tvec'] = ret['cam_from_world'].translation

    if not ret['success']:
        ret['mkpq'] = mkpq
        ret['3D_ids'] = all_3D_ids
        ret['db_ids'] = db_ids
        ret['score_q'] = all_score_q
        ret['log_info'] = log_info
        ret['qvec'] = qvec
        ret['tvec'] = tvec
        ret['inliers'] = [False for i in range(mkpq.shape[0])]
        ret['num_inliers'] = 0
        ret['keypoints_query'] = np.array([])
        ret['points3D_ids'] = []
        return ret

    ret_inliers = ret['inliers']
    loc_keypoints_query = np.array([mkpq[i] for i in range(len(ret_inliers)) if ret_inliers[i]])
    loc_points3D_ids = [all_3D_ids[i] for i in range(len(ret_inliers)) if ret_inliers[i]]

    ret['mkpq'] = mkpq
    ret['3D_ids'] = all_3D_ids
    ret['db_ids'] = db_ids
    ret['log_info'] = log_info
    ret['keypoints_query'] = loc_keypoints_query
    ret['points3D_ids'] = loc_points3D_ids

    return ret


# proposed in efficient large-scale localization by global instance recognition, cvpr 2022
def pose_estimator_iterative(qname, qinfo, db_ids, db_images, points3D, feature_file, thresh, image_dir,
                             matcher,
                             inlier_th=50,
                             log_info=None,
                             do_covisibility_opt=False,
                             covisibility_frame=50,
                             vis_dir=None,
                             obs_th=0,
                             opt_th=12,
                             gt_qvec=None,
                             gt_tvec=None,
                             query_img_prefix='',
                             db_img_prefix='',
                             ):
    print("qname: ", qname)
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    # q_img = cv2.imread(osp.join(image_dir, query_img_prefix, qname))

    kpq = feature_file[qname]['keypoints'][()]
    score_q = feature_file[qname]['scores'][()]
    imgsize_q = feature_file[qname]['image_size'][()]
    desc_q = feature_file[qname]['descriptors'][()]
    desc_q = desc_q.transpose()  # [N D]
    query_data = {
        'keypoints': kpq,
        'scores': score_q,
        'descriptors': desc_q,
        'image_size': imgsize_q,
    }
    camera_model, width, height, params = qinfo

    best_results = {
        'tvec': None,
        'qvec': None,
        'num_inliers': 0,
        'single_num_inliers': 0,
        'db_id': -1,
        'order': -1,
        'qname': qname,
        'optimize': False,
        'dbname': db_images[db_ids[0]].name,
        "ret_source": "",
        "inliers": [],
        'keypoints_query': np.array([]),
        'points3D_ids': [],
    }

    cam = pycolmap.Camera(model=camera_model, width=width, height=height, params=params)

    for cluster_idx, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        mp3d, mkpq, mp3d_ids, q_ids = find_2D_3D_matches(
            query_data=query_data,
            db_id=db_id,
            points3D=points3D,
            feature_file=feature_file,
            db_images=db_images,
            matcher=matcher,
            obs_th=obs_th)

        if mp3d.shape[0] < 8:
            print_text = "qname: {:s} dbname: {:s}({:d}/{:d}) failed because of insufficient 3d points {:d}".format(
                qname,
                db_name,
                cluster_idx + 1,
                len(db_ids),
                mp3d.shape[0])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cam,
                                                estimation_options={
                                                    "ransac": {"max_error": thresh}},
                                                refinement_options={},
                                                )

        if ret is None:
            ret = {'success': False, }
        else:
            ret['success'] = True
            ret['qvec'] = ret['cam_from_world'].rotation.quat[[3, 0, 1, 2]]
            ret['tvec'] = ret['cam_from_world'].translation

        if not ret["success"]:
            print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed after matching".format(qname, db_name,
                                                                                             cluster_idx + 1,
                                                                                             len(db_ids))
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        inliers = ret['inliers']
        num_inliers = ret['num_inliers']
        inlier_p3d_ids = [mp3d_ids[i] for i in range(len(inliers)) if inliers[i]]
        inlier_mkpq = [mkpq[i] for i in range(len(inliers)) if inliers[i]]
        loc_keypoints_query = np.array(inlier_mkpq)
        loc_points3D_ids = inlier_p3d_ids

        if ret['num_inliers'] > best_results['num_inliers']:
            best_results['qvec'] = ret['qvec']
            best_results['tvec'] = ret['tvec']
            best_results['inlier'] = ret['inliers']
            best_results['num_inliers'] = ret['num_inliers']
            best_results['dbname'] = db_name
            best_results['order'] = cluster_idx + 1
            best_results['keypoints_query'] = loc_keypoints_query
            best_results['points3D_ids'] = loc_points3D_ids

        if ret['num_inliers'] < inlier_th:
            print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed insufficient {:d} inliers".format(qname,
                                                                                                        db_name,
                                                                                                        cluster_idx + 1,
                                                                                                        len(db_ids),
                                                                                                        num_inliers,
                                                                                                        )
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) initialization succeed with {:d} inliers".format(
            qname,
            db_name,
            cluster_idx + 1,
            len(db_ids),
            ret["num_inliers"]
        )
        print(print_text)
        if log_info is not None:
            log_info += (print_text + '\n')

        if do_covisibility_opt:
            ret = pose_refinement(qname=qname,
                                  query_cam=cam,
                                  feature_file=feature_file,
                                  db_frame_id=db_id,
                                  db_images=db_images,
                                  points3D=points3D,
                                  thresh=thresh,
                                  covisibility_frame=covisibility_frame,
                                  matcher=matcher,
                                  obs_th=obs_th,
                                  opt_th=opt_th,
                                  qvec=ret['qvec'],
                                  tvec=ret['tvec'],
                                  log_info='',
                                  image_dir=image_dir,
                                  vis_dir=vis_dir,
                                  gt_qvec=gt_qvec,
                                  gt_tvec=gt_tvec,
                                  )

            loc_keypoints_query = ret['keypoints_query']
            loc_points3D_ids = ret['points3D_ids']

            log_info = log_info + ret['log_info']
            print_text = 'Find {:d} inliers after optimization'.format(ret['num_inliers'])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + "\n")

        # localization succeed
        qvec = ret['qvec']
        tvec = ret['tvec']
        num_inliers = ret['num_inliers']
        best_results['keypoints_query'] = loc_keypoints_query
        best_results['points3D_ids'] = loc_points3D_ids

        best_results['qvec'] = qvec
        best_results['tvec'] = tvec
        best_results['num_inliers'] = num_inliers
        best_results['log_info'] = log_info

        return best_results

    if best_results['num_inliers'] >= 10:  # 20 for aachen
        qvec = best_results['qvec']
        tvec = best_results['tvec']
        best_dbname = best_results['dbname']

        best_results['keypoints_query'] = loc_keypoints_query
        best_results['points3D_ids'] = loc_points3D_ids

        if do_covisibility_opt:
            ret = pose_refinement(qname=qname,
                                  query_cam=cam,
                                  feature_file=feature_file,
                                  db_frame_id=db_name_to_id[best_dbname],
                                  db_images=db_images,
                                  points3D=points3D,
                                  thresh=thresh,
                                  covisibility_frame=covisibility_frame,
                                  matcher=matcher,
                                  obs_th=obs_th,
                                  opt_th=opt_th,
                                  qvec=qvec,
                                  tvec=tvec,
                                  log_info='',
                                  image_dir=image_dir,
                                  vis_dir=vis_dir,
                                  gt_qvec=gt_qvec,
                                  gt_tvec=gt_tvec,
                                  )

        # localization succeed
        qvec = ret['qvec']
        tvec = ret['tvec']
        num_inliers = ret['num_inliers']
        best_results['keypoints_query'] = loc_keypoints_query
        best_results['points3D_ids'] = loc_points3D_ids

        best_results['qvec'] = qvec
        best_results['tvec'] = tvec
        best_results['num_inliers'] = num_inliers
        best_results['log_info'] = log_info

        return best_results

    closest = db_images[db_ids[0][0]]
    print_text = 'Localize {:s} failed, but use the pose of {:s} as approximation'.format(qname, closest.name)
    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')

    best_results['qvec'] = closest.qvec
    best_results['tvec'] = closest.tvec
    best_results['num_inliers'] = -1
    best_results['log_info'] = log_info

    return best_results
