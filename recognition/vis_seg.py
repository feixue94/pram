# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> vis_seg
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 11:06
=================================================='''
import cv2
import numpy as np
from copy import deepcopy


def myHash(text: str):
    hash = 0
    for ch in text:
        hash = (hash * 7879 ^ ord(ch) * 5737) & 0xFFFFFFFF
    return hash


def generate_color_dic(n_seg=1000):
    out = {}
    for i in range(n_seg + 1):
        sid = i
        if sid == 0:
            color = (0, 0, 255)  # [b, g, r]
        else:
            # rgb_new = hash(str(sid * 319993))
            rgb_new = myHash(str(sid * 319993))
            r = (rgb_new & 0xFF0000) >> 16
            g = (rgb_new & 0x00FF00) >> 8
            b = rgb_new & 0x0000FF
            color = (b, g, r)
        out[i] = color
    return out


def vis_seg_point(img, kpts, segs=None, seg_color=None, radius=7, thickness=-1):
    outimg = deepcopy(img)
    for i in range(kpts.shape[0]):
        # print(kpts[i])
        if segs is not None and seg_color is not None:
            color = seg_color[segs[i]]
        else:
            color = (0, 255, 0)
        outimg = cv2.circle(outimg,
                            center=(int(kpts[i, 0]), int(kpts[i, 1])),
                            color=color,
                            radius=radius,
                            thickness=thickness, )

    return outimg


def vis_corr_incorr_point(img, kpts, pred_segs, gt_segs, radius=7, thickness=-1):
    outimg = deepcopy(img)
    for i in range(kpts.shape[0]):
        # print(kpts[i])
        p_seg = pred_segs[i]
        g_seg = gt_segs[i]
        if p_seg == g_seg:
            if g_seg != 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        outimg = cv2.circle(outimg,
                            center=(int(kpts[i, 0]), int(kpts[i, 1])),
                            color=color,
                            radius=radius,
                            thickness=thickness, )
    return outimg


def vis_inlier(img, kpts, inliers, radius=7, thickness=1, with_outlier=True):
    outimg = deepcopy(img)
    for i in range(kpts.shape[0]):
        if not with_outlier:
            if not inliers[i]:
                continue
        if inliers[i]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        outimg = cv2.rectangle(outimg,
                               pt1=(int(kpts[i, 0] - radius), int(kpts[i, 1] - radius)),
                               pt2=(int(kpts[i, 0] + radius), int(kpts[i, 1] + radius)),
                               color=color,
                               thickness=thickness, )

    return outimg


def vis_global_seg(cls, seg_color, radius=7, thickness=-1):
    all_patches = []
    for i in range(cls.shape[0]):
        if cls[i] == 0:
            continue
        color = seg_color[i]
        patch = np.zeros(shape=(radius, radius, 3), dtype=np.uint8)
        patch[..., 0] = color[0]
        patch[..., 1] = color[1]
        patch[..., 2] = color[2]

        all_patches.append(patch)
    if len(all_patches) == 0:
        color = seg_color[0]
        patch = np.zeros(shape=(radius, radius, 3), dtype=np.uint8)
        patch[..., 0] = color[0]
        patch[..., 1] = color[1]
        patch[..., 2] = color[2]
        all_patches.append(patch)
    return np.vstack(all_patches)


def plot_matches(img1, img2, pts1, pts2, inliers, radius=3, line_thickness=2, horizon=True, plot_outlier=False,
                 confs=None):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    # r = 3
    if horizon:
        img_out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
        # Place the first image to the left
        img_out[:rows1, :cols1] = img1
        # Place the next image to the right of it
        img_out[:rows2, cols1:] = img2  # np.dstack([img2, img2, img2])
        for idx in range(inliers.shape[0]):
            # if idx % 10 > 0:
            #     continue
            if inliers[idx]:
                color = (0, 255, 0)
            else:
                if not plot_outlier:
                    continue
                color = (0, 0, 255)
            pt1 = pts1[idx]
            pt2 = pts2[idx]

            if confs is not None:
                nr = int(radius * confs[idx])
            else:
                nr = radius
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), nr, color, 2)

            img_out = cv2.circle(img_out, (int(pt2[0]) + cols1, int(pt2[1])), nr, color, 2)

            img_out = cv2.line(img_out, (int(pt1[0]), int(pt1[1])), (int(pt2[0]) + cols1, int(pt2[1])), color,
                               line_thickness)
    else:
        img_out = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
        # Place the first image to the left
        img_out[:rows1, :cols1] = img1
        # Place the next image to the right of it
        img_out[rows1:, :cols2] = img2  # np.dstack([img2, img2, img2])

        for idx in range(inliers.shape[0]):
            # print("idx: ", inliers[idx])
            # if idx % 10 > 0:
            #     continue
            if inliers[idx]:
                color = (0, 255, 0)
            else:
                if not plot_outlier:
                    continue
                color = (0, 0, 255)

            if confs is not None:
                nr = int(radius * confs[idx])
            else:
                nr = radius

            pt1 = pts1[idx]
            pt2 = pts2[idx]
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), nr, color, 2)

            img_out = cv2.circle(img_out, (int(pt2[0]), int(pt2[1]) + rows1), nr, color, 2)

            img_out = cv2.line(img_out, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]) + rows1), color,
                               line_thickness)

    return img_out


def plot_kpts(img, kpts, radius=None, colors=None, r=3, color=(0, 0, 255), nh=-1, nw=-1, shape='o', show_text=None,
              thickness=5):
    img_out = deepcopy(img)
    for i in range(kpts.shape[0]):
        pt = kpts[i]
        if radius is not None:
            if shape == 'o':
                img_out = cv2.circle(img_out, center=(int(pt[0]), int(pt[1])), radius=radius[i],
                                     color=color if colors is None else colors[i],
                                     thickness=thickness)
            elif shape == '+':
                img_out = cv2.line(img_out, pt1=(int(pt[0] - radius[i]), int(pt[1])),
                                   pt2=(int(pt[0] + radius[i]), int(pt[1])),
                                   color=color if colors is None else colors[i],
                                   thickness=5)
                img_out = cv2.line(img_out, pt1=(int(pt[0]), int(pt[1] - radius[i])),
                                   pt2=(int(pt[0]), int(pt[1] + radius[i])), color=color,
                                   thickness=thickness)
        else:
            if shape == 'o':
                img_out = cv2.circle(img_out, center=(int(pt[0]), int(pt[1])), radius=r,
                                     color=color if colors is None else colors[i],
                                     thickness=thickness)
            elif shape == '+':
                img_out = cv2.line(img_out, pt1=(int(pt[0] - r), int(pt[1])),
                                   pt2=(int(pt[0] + r), int(pt[1])), color=color if colors is None else colors[i],
                                   thickness=thickness)
                img_out = cv2.line(img_out, pt1=(int(pt[0]), int(pt[1] - r)),
                                   pt2=(int(pt[0]), int(pt[1] + r)), color=color if colors is None else colors[i],
                                   thickness=thickness)

    if show_text is not None:
        img_out = cv2.putText(img_out, show_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                              (0, 0, 255), 3)
    if nh == -1 and nw == -1:
        return img_out
    if nh > 0:
        return cv2.resize(img_out, dsize=(int(img.shape[1] / img.shape[0] * nh), nh))
    if nw > 0:
        return cv2.resize(img_out, dsize=(nw, int(img.shape[0] / img.shape[1] * nw)))
