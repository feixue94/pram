# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> utils
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:31
=================================================='''
import torch


def normalize_size(x, size, scale=0.7):
    size = size.reshape([1, 2])
    norm_fac = size.max() + 0.5
    return (x - size / 2) / (norm_fac * scale)


def collect_batch(batch):
    out = {}
    # if len(batch) == 0:
    #     return batch
    # else:
    for k in batch[0].keys():
        tmp = []
        for v in batch:
            tmp.append(v[k])
        if isinstance(batch[0][k], str) or isinstance(batch[0][k], list):
            out[k] = tmp
        else:
            out[k] = torch.cat([torch.from_numpy(i)[None] for i in tmp], dim=0)

    return out
