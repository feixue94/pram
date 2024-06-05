# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   localizer -> image_to_video
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/09/2023 20:15
=================================================='''
import cv2
import os
import os.path as osp

import numpy as np
from tqdm import tqdm
import argparse

from tools.common import resize_img

parser = argparse.ArgumentParser(description='Image2Video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_dir', type=str, required=True)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--height', type=int, default=-1)
parser.add_argument('--fps', type=int, default=30)


def imgs2video(img_dir, video_path, fps=30, height=1024):
    img_fns = os.listdir(img_dir)
    # print(img_fns)
    img_fns = [v for v in img_fns if v.split('.')[-1] in ['jpg', 'png']]
    img_fns = sorted(img_fns)
    # print(img_fns)
    # 输出视频路径
    # fps = 1

    img = cv2.imread(osp.join(img_dir, img_fns[0]))
    if height == -1:
        height = img.shape[1]
    new_img = resize_img(img=img, nh=height)
    img_size = (new_img.shape[1], height)

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
    # fourcc = cv2.VideoWriter_fourcc('I','4','2','0')

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 设置输出视频为mp4格式
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)

    for i in tqdm(range(0, len(img_fns)), total=len(img_fns)):
        # fn = img_fns[i].split('-')
        im_name = os.path.join(img_dir, img_fns[i])
        print(im_name)
        frame = cv2.imread(im_name, 1)
        # frame = np.flip(frame, 0)

        frame = cv2.resize(frame, dsize=img_size)
        # print(frame.shape)
        # exit(0)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        videoWriter.write(frame)

    videoWriter.release()


if __name__ == '__main__':
    args = parser.parse_args()
    imgs2video(img_dir=args.image_dir, video_path=args.video_path, fps=args.fps, height=args.height)
