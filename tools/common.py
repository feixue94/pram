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
import numpy as np
from typing import Tuple


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


def resize_image_with_padding(image: np.array, nw: int, nh: int, padding_color: Tuple[int] = (0, 0, 0)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])  # (w, h)
    ratio_w = nw / original_shape[0]
    ratio_h = nh / original_shape[1]

    if ratio_w == ratio_h:
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_NEAREST)

    ratio = ratio_w if ratio_w < ratio_h else ratio_h

    new_size = tuple([int(x * ratio) for x in original_shape])
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    delta_w = nw - new_size[0] if nw > new_size[0] else new_size[0] - nw
    delta_h = nh - new_size[1] if nh > new_size[1] else new_size[1] - nh

    left, right = delta_w // 2, delta_w - (delta_w // 2)
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)

    # print('top, bottom, left, right: ', top, bottom, left, right)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


def puttext_with_background(image, text, org=(0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, text_color=(0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA, bg_color=None):
    if bg_color is not None:
        (text_width, text_height), baseline = cv2.getTextSize(text,
                                                              fontFace,
                                                              fontScale=fontScale,
                                                              thickness=thickness)
        box_coords = (
            (org[0], org[1] + baseline),
            (org[0] + text_width + 2, org[1] - text_height - 2))

        cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    out_img = cv2.putText(img=image, text=text,
                          org=org,
                          fontFace=fontFace,
                          fontScale=fontScale, color=text_color,
                          thickness=thickness, lineType=lineType)
    return out_img
