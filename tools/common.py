# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> common
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 15:05
=================================================='''
import os
import torch
import json
import yaml
import cv2


def load_args(args, save_path):
    with open(save_path, "r") as f:
        args.__dict__ = json.load(f)


def save_args_yaml(args, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(args, f)


def merge_tags(tags: list, connection='_'):
    out = ''
    for i, t in enumerate(tags):
        if i == 0:
            out = out + t
        else:
            out = out + connection + t
    return out


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        # print(os.environ['CUDA_VISIBLE_DEVICES'])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True  # speed-up cudnn
        torch.backends.cudnn.fastest = True  # even more speed-up?
        print('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        print('Launching on CPU')

    return cuda


def resize_img(img, nh=-1, nw=-1, rmax=-1, mode=cv2.INTER_NEAREST):
    assert nh > 0 or nw > 0 or rmax > 0
    if nh > 0:
        return cv2.resize(img, dsize=(int(img.shape[1] / img.shape[0] * nh), nh), interpolation=mode)
    if nw > 0:
        return cv2.resize(img, dsize=(nw, int(img.shape[0] / img.shape[1] * nw)), interpolation=mode)
    if rmax > 0:
        oh, ow = img.shape[0], img.shape[1]
        if oh > ow:
            return cv2.resize(img, dsize=(int(img.shape[1] / img.shape[0] * rmax), rmax), interpolation=mode)
        else:
            return cv2.resize(img, dsize=(rmax, int(img.shape[0] / img.shape[1] * rmax)), interpolation=mode)

    return cv2.resize(img, dsize=(nw, nh), interpolation=mode)
