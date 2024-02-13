# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> sfd2
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 14:53
=================================================='''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as tvf

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.Normalize(mean=RGB_mean, std=RGB_std)])


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', align_corners=True)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=False, groups=1, dilation=1):
    if not use_bn:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, groups=32, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, outplanes)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes, stride, groups, dilation)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = conv1x1(outplanes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet4x(nn.Module):
    default_config = {
        'conf_th': 0.005,
        'remove_borders': 4,
        'min_keypoints': 128,
        'max_keypoints': 4096,
    }

    def __init__(self, inputdim=3, outdim=128, desc_compressor=None):
        super().__init__()
        self.outdim = outdim
        self.desc_compressor = desc_compressor

        d1, d2, d3, d4, d5, d6 = 64, 128, 256, 256, 256, 256
        self.conv1a = conv(in_channels=inputdim, out_channels=d1, kernel_size=3, use_bn=True)
        self.conv1b = conv(in_channels=d1, out_channels=d1, kernel_size=3, stride=2, use_bn=True)

        self.conv2a = conv(in_channels=d1, out_channels=d2, kernel_size=3, use_bn=True)
        self.conv2b = conv(in_channels=d2, out_channels=d2, kernel_size=3, stride=2, use_bn=True)

        self.conv3a = conv(in_channels=d2, out_channels=d3, kernel_size=3, use_bn=True)
        self.conv3b = conv(in_channels=d3, out_channels=d3, kernel_size=3, use_bn=True)

        self.conv4 = nn.Sequential(
            ResBlock(inplanes=256, outplanes=256, groups=32),
            ResBlock(inplanes=256, outplanes=256, groups=32),
            ResBlock(inplanes=256, outplanes=256, groups=32),
        )

        self.convPa = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.convDa = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )

        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.convDb = torch.nn.Conv2d(256, outdim, kernel_size=1, stride=1, padding=0)

    def det(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(out1a)

        out2a = self.conv2a(out1b)
        out2b = self.conv2b(out2a)

        out3a = self.conv3a(out2b)
        out3b = self.conv3b(out3a)

        out4 = self.conv4(out3b)

        cPa = self.convPa(out4)
        logits = self.convPb(cPa)
        full_semi = torch.softmax(logits, dim=1)
        semi = full_semi[:, :-1, :, :]
        Hc, Wc = semi.size(2), semi.size(3)
        score = semi.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)

        # Descriptor Head
        cDa = self.convDa(out4)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        return score, desc

    def forward(self, batch):
        out1a = self.conv1a(batch['image'])
        out1b = self.conv1b(out1a)

        out2a = self.conv2a(out1b)
        out2b = self.conv2b(out2a)

        out3a = self.conv3a(out2b)
        out3b = self.conv3b(out3a)

        out4 = self.conv4(out3b)

        cPa = self.convPa(out4)
        logits = self.convPb(cPa)
        full_semi = torch.softmax(logits, dim=1)
        semi = full_semi[:, :-1, :, :]
        Hc, Wc = semi.size(2), semi.size(3)
        score = semi.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)

        # Descriptor Head
        cDa = self.convDa(out4)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        return {
            'dense_features': desc,
            'scores': score,
            'logits': logits,
            'semi_map': semi,
        }

    def extract_patches(self, batch):
        out1a = self.conv1a(batch['image'])
        out1b = self.conv1b(out1a)

        out2a = self.conv2a(out1b)
        out2b = self.conv2b(out2a)

        out3a = self.conv3a(out2b)
        out3b = self.conv3b(out3a)

        out4 = self.conv4(out3b)

        cPa = self.convPa(out4)
        logits = self.convPb(cPa)
        full_semi = torch.softmax(logits, dim=1)
        semi = full_semi[:, :-1, :, :]
        Hc, Wc = semi.size(2), semi.size(3)
        score = semi.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)

        # Descriptor Head
        cDa = self.convDa(out4)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        return {
            'dense_features': desc,
            'scores': score,
            'logits': logits,
            'semi_map': semi,
        }

    def extract_local_global(self, data,
                             config={
                                 'conf_th': 0.005,
                                 'remove_borders': 4,
                                 'min_keypoints': 128,
                                 'max_keypoints': 4096,
                             }
                             ):

        config = {**self.default_config, **config}

        b, ic, ih, iw = data['image'].shape
        out1a = self.conv1a(data['image'])
        out1b = self.conv1b(out1a)  # 64

        out2a = self.conv2a(out1b)
        out2b = self.conv2b(out2a)  # 128

        out3a = self.conv3a(out2b)
        out3b = self.conv3b(out3a)  # 256

        out4 = self.conv4(out3b)  # 256

        cPa = self.convPa(out4)
        logits = self.convPb(cPa)
        full_semi = torch.softmax(logits, dim=1)
        semi = full_semi[:, :-1, :, :]
        Hc, Wc = semi.size(2), semi.size(3)
        score = semi.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)
        if Hc * 8 != ih or Wc * 8 != iw:
            score = F.interpolate(score.unsqueeze(1), size=[ih, iw], align_corners=True, mode='bilinear')
            score = score.squeeze(1)
        # extract keypoints
        nms_scores = simple_nms(scores=score, nms_radius=4)
        keypoints = [
            torch.nonzero(s >= config['conf_th'])
            for s in nms_scores]
        scores = [s[tuple(k.t())] for s, k in zip(nms_scores, keypoints)]

        if len(scores[0]) <= config['min_keypoints']:
            keypoints = [
                torch.nonzero(s >= config['conf_th'] * 0.5)
                for s in nms_scores]
            scores = [s[tuple(k.t())] for s, k in zip(nms_scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, config['remove_borders'], ih, iw)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        # Descriptor Head
        cDa = self.convDa(out4)
        desc_map = self.convDb(cDa)
        desc_map = F.normalize(desc_map, dim=1)

        descriptors = [sample_descriptors(k[None], d[None], 4)[0]
                       for k, d in zip(keypoints, desc_map)]

        return {
            'score_map': score,
            'desc_map': desc_map,
            'mid_features': out4,
            'global_descriptors': [out1b, out2b, out3b, out4],
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }

    def sample(self, score_map, semi_descs, kpts, s=4, norm_desc=True):
        # print('sample: ', score_map.shape, semi_descs.shape, kpts.shape)
        b, c, h, w = semi_descs.shape
        norm_kpts = kpts - s / 2 + 0.5
        norm_kpts = norm_kpts / torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                                             ).to(norm_kpts)[None]
        norm_kpts = norm_kpts * 2 - 1
        # args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            semi_descs, norm_kpts.view(b, 1, -1, 2), mode='bilinear', align_corners=True)

        if norm_desc:
            descriptors = torch.nn.functional.normalize(
                descriptors.reshape(b, c, -1), p=2, dim=1)
        else:
            descriptors = descriptors.reshape(b, c, -1)

        # print('max: ', torch.min(kpts[:, 1].long()), torch.max(kpts[:, 1].long()), torch.min(kpts[:, 0].long()),
        #       torch.max(kpts[:, 0].long()))
        scores = score_map[0, kpts[:, 1].long(), kpts[:, 0].long()]

        return scores, descriptors.squeeze(0)


class DescriptorCompressor(nn.Module):
    def __init__(self, inputdim: int, outdim: int):
        super().__init__()
        self.inputdim = inputdim
        self.outdim = outdim
        self.conv = nn.Conv1d(in_channels=inputdim, out_channels=outdim, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # b, c, n = x.shape
        out = self.conv(x)
        out = F.normalize(out, p=2, dim=1)
        return out


def extract_sfd2_return(model, img, conf_th=0.001,
                        mask=None,
                        topK=-1,
                        min_keypoints=0,
                        **kwargs):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    img = norm_RGB(img.squeeze())
    img = img[None]
    img = img.cuda()

    B, one, H, W = img.shape

    all_pts = []
    all_descs = []

    if 'scales' in kwargs.keys():
        scales = kwargs.get('scales')
    else:
        scales = [1.0]

    for s in scales:
        if s == 1.0:
            new_img = img
        else:
            nh = int(H * s)
            nw = int(W * s)
            new_img = F.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=True)
        nh, nw = new_img.shape[2:]

        with torch.no_grad():
            heatmap, coarse_desc = model.det(new_img)

            # print("nh, nw, heatmap, desc: ", nh, nw, heatmap.shape, coarse_desc.shape)
            if len(heatmap.size()) == 3:
                heatmap = heatmap.unsqueeze(1)
            if len(heatmap.size()) == 2:
                heatmap = heatmap.unsqueeze(0)
                heatmap = heatmap.unsqueeze(1)
            # print(heatmap.shape)
            if heatmap.size(2) != nh or heatmap.size(3) != nw:
                heatmap = F.interpolate(heatmap, size=[nh, nw], mode='bilinear', align_corners=True)

            conf_thresh = conf_th
            nms_dist = 3
            border_remove = 4
            scores = simple_nms(heatmap, nms_radius=nms_dist)
            keypoints = [
                torch.nonzero(s > conf_thresh)
                for s in scores]
            scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
            # print('scores in return: ', len(scores[0]))

            # print(keypoints[0].shape)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]
            scores = scores[0].data.cpu().numpy().squeeze()
            keypoints = keypoints[0].data.cpu().numpy().squeeze()
            pts = keypoints.transpose()
            pts[2, :] = scores

            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Sort by confidence.
            # Remove points along border.
            bord = border_remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]

            # valid_idex = heatmap > conf_thresh
            # valid_score = heatmap[valid_idex]
            # """
            # --- Process descriptor.
            # coarse_desc = coarse_desc.data.cpu().numpy().squeeze()
            D = coarse_desc.size(1)
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                if coarse_desc.size(2) == nh and coarse_desc.size(3) == nw:
                    desc = coarse_desc[:, :, pts[1, :], pts[0, :]]
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                else:
                    # Interpolate into descriptor map using 2D point locations.
                    samp_pts = torch.from_numpy(pts[:2, :].copy())
                    samp_pts[0, :] = (samp_pts[0, :] / (float(nw) / 2.)) - 1.
                    samp_pts[1, :] = (samp_pts[1, :] / (float(nh) / 2.)) - 1.
                    samp_pts = samp_pts.transpose(0, 1).contiguous()
                    samp_pts = samp_pts.view(1, 1, -1, 2)
                    samp_pts = samp_pts.float()
                    samp_pts = samp_pts.cuda()
                    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, mode='bilinear', align_corners=True)
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

            if pts.shape[1] == 0:
                continue

            # print(pts.shape, heatmap.shape, new_img.shape, img.shape, nw, nh, W, H)
            pts[0, :] = pts[0, :] * W / nw
            pts[1, :] = pts[1, :] * H / nh
            all_pts.append(np.transpose(pts, [1, 0]))
            all_descs.append(np.transpose(desc, [1, 0]))

    all_pts = np.vstack(all_pts)
    all_descs = np.vstack(all_descs)

    torch.backends.cudnn.benchmark = old_bm

    if all_pts.shape[0] == 0:
        return None, None, None

    keypoints = all_pts[:, 0:2]
    scores = all_pts[:, 2]
    descriptors = all_descs

    if mask is not None:
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        labels = []
        others = []
        keypoints_with_labels = []
        scores_with_labels = []
        descriptors_with_labels = []
        keypoints_without_labels = []
        scores_without_labels = []
        descriptors_without_labels = []

        id_img = np.int32(mask[:, :, 2]) * 256 * 256 + np.int32(mask[:, :, 1]) * 256 + np.int32(mask[:, :, 0])
        # print(img.shape, id_img.shape)

        for i in range(keypoints.shape[0]):
            x = keypoints[i, 0]
            y = keypoints[i, 1]
            # print("x-y", x, y, int(x), int(y))
            gid = id_img[int(y), int(x)]
            if gid == 0:
                keypoints_without_labels.append(keypoints[i])
                scores_without_labels.append(scores[i])
                descriptors_without_labels.append(descriptors[i])
                others.append(0)
            else:
                keypoints_with_labels.append(keypoints[i])
                scores_with_labels.append(scores[i])
                descriptors_with_labels.append(descriptors[i])
                labels.append(gid)

        if topK > 0:
            if topK <= len(keypoints_with_labels):
                idxes = np.array(scores_with_labels, float).argsort()[::-1][:topK]
                keypoints = np.array(keypoints_with_labels, float)[idxes]
                scores = np.array(scores_with_labels, float)[idxes]
                labels = np.array(labels, np.int32)[idxes]
                descriptors = np.array(descriptors_with_labels, float)[idxes]
            elif topK >= len(keypoints_with_labels) + len(keypoints_without_labels):
                # keypoints = np.vstack([keypoints_with_labels, keypoints_without_labels])
                # scores = np.vstack([scorescc_with_labels, scores_without_labels])
                # descriptors = np.vstack([descriptors_with_labels, descriptors_without_labels])
                # labels = np.vstack([labels, others])
                keypoints = keypoints_with_labels
                scores = scores_with_labels
                descriptors = descriptors_with_labels
                for i in range(len(others)):
                    keypoints.append(keypoints_without_labels[i])
                    scores.append(scores_without_labels[i])
                    descriptors.append(descriptors_without_labels[i])
                    labels.append(others[i])
            else:
                n = topK - len(keypoints_with_labels)
                idxes = np.array(scores_without_labels, float).argsort()[::-1][:n]
                keypoints = keypoints_with_labels
                scores = scores_with_labels
                descriptors = descriptors_with_labels
                for i in idxes:
                    keypoints.append(keypoints_without_labels[i])
                    scores.append(scores_without_labels[i])
                    descriptors.append(descriptors_without_labels[i])
                    labels.append(others[i])
        keypoints = np.array(keypoints, float)
        descriptors = np.array(descriptors, float)
        # print(keypoints.shape, descriptors.shape)
        return {"keypoints": np.array(keypoints, float),
                "descriptors": np.array(descriptors, float),
                "scores": np.array(scores, np.float),
                "labels": np.array(labels, np.int32),
                }
    else:
        # print(topK)
        if topK > 0:
            idxes = np.array(scores, dtype=float).argsort()[::-1][:topK]
            keypoints = np.array(keypoints[idxes], dtype=float)
            scores = np.array(scores[idxes], dtype=float)
            descriptors = np.array(descriptors[idxes], dtype=float)

        keypoints = np.array(keypoints, dtype=float)
        scores = np.array(scores, dtype=float)
        descriptors = np.array(descriptors, dtype=float)

        # print(keypoints.shape, descriptors.shape)

        return {"keypoints": np.array(keypoints, dtype=float),
                "descriptors": descriptors,
                "scores": scores,
                }
