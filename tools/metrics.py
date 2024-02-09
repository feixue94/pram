# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> metrics
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 16:32
=================================================='''
import torch
import numpy as np
import torch.nn.functional as F


class SeqIOU:
    def __init__(self, n_class, ignored_sids=[]):
        self.n_class = n_class
        self.ignored_sids = ignored_sids
        self.class_iou = np.zeros(n_class)
        self.precisions = []

    def add(self, pred, target):
        for i in range(self.n_class):
            inter = np.sum((pred == target) * (target == i))
            union = np.sum(target == i) + np.sum(pred == i) - inter
            if union > 0:
                self.class_iou[i] = inter / union

        acc = (pred == target)
        if len(self.ignored_sids) == 0:
            acc_ratio = np.sum(acc) / pred.shape[0]
        else:
            pred_mask = (pred >= 0)
            target_mask = (target >= 0)
            for i in self.ignored_sids:
                pred_mask = pred_mask & (pred == i)
                target_mask = target_mask & (target == i)

            acc = acc & (1 - pred_mask)
            tgt = (1 - target_mask)
            if np.sum(tgt) == 0:
                acc_ratio = 0
            else:
                acc_ratio = np.sum(acc) / np.sum(tgt)

        self.precisions.append(acc_ratio)

    def get_mean_iou(self):
        return np.mean(self.class_iou)

    def get_mean_precision(self):
        return np.mean(self.precisions)

    def clear(self):
        self.precisions = []
        self.class_iou = np.zeros(self.n_class)


def compute_iou(pred: np.ndarray, target: np.ndarray, n_class: int, ignored_ids=[]) -> float:
    class_iou = np.zeros(n_class)
    for i in range(n_class):
        if i in ignored_ids:
            continue
        inter = np.sum((pred == target) * (target == i))
        union = np.sum(target == i) + np.sum(pred == i) - inter
        if union > 0:
            class_iou[i] = inter / union

    return np.mean(class_iou)
    # return class_iou


def compute_precision(pred: np.ndarray, target: np.ndarray, ignored_ids: list = []) -> float:
    acc = (pred == target)
    if len(ignored_ids) == 0:
        return np.sum(acc) / pred.shape[0]
    else:
        pred_mask = (pred >= 0)
        target_mask = (target >= 0)
        for i in ignored_ids:
            pred_mask = pred_mask & (pred == i)
            target_mask = target_mask & (target == i)

        acc = acc & (1 - pred_mask)
        tgt = (1 - target_mask)
        if np.sum(tgt) == 0:
            return 0
        return np.sum(acc) / np.sum(tgt)


def compute_cls_corr(pred: torch.Tensor, target: torch.Tensor, k: int = 20) -> torch.Tensor:
    bs = pred.shape[0]
    _, target_ids = torch.topk(target, k=k, dim=1)
    target_ids = target_ids.cpu().numpy()
    _, top_ids = torch.topk(pred, k=k, dim=1)  # [B, k, 1]
    top_ids = top_ids.cpu().numpy()
    acc = 0
    for i in range(bs):
        # print('top_ids: ', i, top_ids[i], target_ids[i])
        overlap = [v for v in top_ids[i] if v in target_ids[i] and v >= 0]
        acc = acc + len(overlap) / k
    acc = acc / bs
    return torch.from_numpy(np.array([acc])).to(pred.device)


def compute_corr_incorr(pred: torch.Tensor, target: torch.Tensor, ignored_ids: list = []) -> tuple:
    '''
    :param pred: [B, N, C]
    :param target: [B, N]
    :param ignored_ids: []
    :return:
    '''
    pred_ids = torch.max(pred, dim=-1)[1]
    if len(ignored_ids) == 0:
        acc = (pred_ids == target)
        inacc = torch.logical_not(acc)
        acc_ratio = torch.sum(acc) / torch.numel(target)
        inacc_ratio = torch.sum(inacc) / torch.numel(target)
    else:
        acc = (pred_ids == target)
        inacc = torch.logical_not(acc)

        mask = torch.zeros_like(acc)
        for i in ignored_ids:
            mask = torch.logical_and(mask, (target == i))

        acc = torch.logical_and(acc, torch.logical_not(mask))
        acc_ratio = torch.sum(acc) / torch.numel(target)
        inacc_ratio = torch.sum(inacc) / torch.numel(target)

    return acc_ratio, inacc_ratio


def compute_seg_loss_weight(pred: torch.Tensor, target: torch.Tensor, background_id: int = 0,
                            weight_background: float = 0.1) -> torch.Tensor:
    weight = torch.ones(size=(pred.shape[1],), device=pred.device).float()
    pred = torch.log_softmax(pred, dim=1)
    weight[background_id] = weight_background
    seg_loss = F.cross_entropy(pred, target, weight=weight)
    return seg_loss


def compute_cls_loss_ce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cls_loss = torch.zeros(size=[], device=pred.device)
    if len(pred.shape) == 2:
        n_valid = torch.sum(target > 0)
        cls_loss = cls_loss + torch.nn.functional.cross_entropy(pred, target, reduction='sum')
        cls_loss = cls_loss / n_valid
    else:
        for i in range(pred.shape[-1]):
            cls_loss = cls_loss + torch.nn.functional.cross_entropy(pred[..., i], target[..., i], reduction='sum')
        n_valid = torch.sum(target > 0)
        cls_loss = cls_loss / n_valid

    return cls_loss


def compute_cls_loss_kl(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cls_loss = torch.zeros(size=[], device=pred.device)
    if len(pred.shape) == 2:
        cls_loss = cls_loss + torch.nn.functional.kl_div(torch.log_softmax(pred, dim=-1),
                                                         torch.softmax(target, dim=-1),
                                                         reduction='sum')
    else:
        for i in range(pred.shape[-1]):
            cls_loss = cls_loss + torch.nn.functional.kl_div(torch.log_softmax(pred[..., i], dim=-1),
                                                             torch.softmax(target[..., i], dim=-1),
                                                             reduction='sum')

        cls_loss = cls_loss / pred.shape[-1]

    return cls_loss


def compute_sc_loss_l1(pred: torch.Tensor, target: torch.Tensor, mean_xyz=None, scale_xyz=None, mask=None):
    '''
    :param pred: [B, N, C]
    :param target: [B, N, C]
    :param mean_xyz:
    :param scale_xyz:
    :param mask:
    :return:
    '''
    loss = (pred - target)
    loss = torch.abs(loss).mean(dim=1)
    if mask is not None:
        return torch.mean(loss[mask])
    else:
        return torch.mean(loss)


def compute_sc_loss_geo(pred: torch.Tensor, P, K, p2ds, mean_xyz, scale_xyz, max_value=20, mask=None):
    b, c, n = pred.shape
    p3ds = (pred * scale_xyz[..., None].repeat(1, 1, n) + mean_xyz[..., None].repeat(1, 1, n))
    p3ds_homo = torch.cat(
        [pred, torch.ones(size=(p3ds.shape[0], 1, p3ds.shape[2]), dtype=p3ds.dtype, device=p3ds.device)],
        dim=1)  # [B, 4, N]
    p3ds = torch.matmul(K, torch.matmul(P, p3ds_homo)[:, :3, :])  # [B, 3, N]
    # print('p3ds: ', p3ds.shape, P.shape, K.shape, p2ds.shape)

    p2ds_ = p3ds[:, :2, :] / p3ds[:, 2:, :]

    loss = ((p2ds_ - p2ds.permute(0, 2, 1)) ** 2).sum(1)
    loss = torch.clamp_max(loss, max=max_value)
    if mask is not None:
        return torch.mean(loss[mask])
    else:
        return torch.mean(loss)
