# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   localizer -> video_to_image
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   13/01/2024 15:29
=================================================='''
import argparse
import os
import os.path as osp
import cv2

parser = argparse.ArgumentParser(description='Image2Video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--height', type=int, default=-1)
parser.add_argument('--sample_ratio', type=int, default=-1)


def main(args):
    video = cv2.VideoCapture(args.video_path)
    nframe = 0
    while True:
        ret, frame = video.read()
        if ret:
            if args.sample_ratio > 0:
                if nframe % args.sample_ratio != 0:
                    nframe += 1
                    continue
            cv2.imwrite(osp.join(args.image_path, '{:06d}.png'.format(nframe)), frame)
            nframe += 1
        else:
            break


if __name__ == '__main__':
    args = parser.parse_args()
    main(args=args)
