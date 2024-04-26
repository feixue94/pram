# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   localizer -> camera_intrinsics
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   15/08/2023 12:33
=================================================='''
import numpy as np


def intrinsics_from_camera(camera_model, params):
    if camera_model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif camera_model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
    else:
        raise Exception("Camera model not supported")

    # intrinsics
    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K
