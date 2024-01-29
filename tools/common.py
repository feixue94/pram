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
