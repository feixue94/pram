# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> utils
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 10:48
=================================================='''
import torch

eps = 1e-8


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]
