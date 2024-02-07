# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> utils
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 15:27
=================================================='''
import numpy as np
from colmap_utils.read_write_model import qvec2rotmat, rotmat2qvec


def read_query_info(query_fn: str, name_prefix='') -> dict:
    results = {}
    with open(query_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            name, camera_model, width, height = l[:4]
            params = np.array(l[4:], float)
            info = (camera_model, int(width), int(height), params)
            results[name_prefix + name] = info
    print('Load {} query images'.format(len(results.keys())))
    return results


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def compute_pose_error(pred_qcw, pred_tcw, gt_qcw, gt_tcw):
    pred_Rcw = qvec2rotmat(qvec=pred_qcw)
    pred_tcw = np.array(pred_tcw, float).reshape(3, 1)
    pred_Rwc = pred_Rcw.transpose()
    pred_twc = -pred_Rcw.transpose() @ pred_tcw

    gt_Rcw = qvec2rotmat(gt_qcw)
    gt_tcw = np.array(gt_tcw, float).reshape(3, 1)
    gt_Rwc = gt_Rcw.transpose()
    gt_twc = -gt_Rcw.transpose() @ gt_tcw

    t_error_xyz = pred_twc - gt_twc
    t_error = np.sqrt(np.sum(t_error_xyz ** 2))

    # pred_qwc = sciR.from_quat([pred_qcw[1], pred_qcw[2], pred_qcw[3], pred_qcw[0]]).as_quat()
    # gt_qwc = sciR.from_quat([gt_qcw[1], gt_qcw[2], gt_qcw[3], gt_qcw[0]]).as_quat()
    #
    # q_error = quaternion_angular_error(q1=pred_qwc, q2=gt_qwc)
    q_error = quaternion_angular_error(q1=pred_qcw, q2=gt_qcw)

    return q_error, t_error
