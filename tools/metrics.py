# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> metrics
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 16:32
=================================================='''
import torch
import numpy as np


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


def compute_iou(pred, target, n_class: int, ignored_ids=[]):
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


def compute_precision(pred: np.ndarray, target: np.ndarray, ignored_ids: list = []):
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


def compute_cls_corr(pred, target, k=20):
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


def compute_corr_incorr(pred: torch.Tensor, target: torch.Tensor, ignored_ids: list = []):
    pred_ids = torch.max(pred, dim=1)[1]
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
