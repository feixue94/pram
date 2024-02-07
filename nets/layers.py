# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> layers
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:46
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange


def MLP(channels: list, do_bn=True, ac_fn='relu', norm_fn='bn'):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if norm_fn == 'in':
                layers.append(nn.InstanceNorm1d(channels[i], eps=1e-3))
            elif norm_fn == 'bn':
                layers.append(nn.BatchNorm1d(channels[i], eps=1e-3))
            if ac_fn == 'relu':
                layers.append(nn.ReLU())
            elif ac_fn == 'gelu':
                layers.append(nn.GELU())
            elif ac_fn == 'lrelu':
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            # if norm_fn == 'ln':
            #     layers.append(nn.LayerNorm(channels[i]))
    return nn.Sequential(*layers)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, M=None):
        '''
        :param query: [B, D, N]
        :param key: [B, D, M]
        :param value: [B, D, M]
        :param M: [B, N, M]
        :return:
        '''

        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]  # [B, D, NH, N]
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5

        if M is not None:
            # print('M: ', scores.shape, M.shape, torch.sum(M, dim=2))
            # scores = scores * M[:, None, :, :].expand_as(scores)
            # with torch.no_grad():
            mask = (1 - M[:, None, :, :]).repeat(1, scores.shape[1], 1, 1).bool()  # [B, H, N, M]
            scores = scores.masked_fill(mask, -torch.finfo(scores.dtype).max)
            prob = F.softmax(scores, dim=-1)  # * (~mask).float()  # * mask.float()
        else:
            prob = F.softmax(scores, dim=-1)

        x = torch.einsum('bhnm,bdhm->bdhn', prob, value)
        self.prob = prob

        out = self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))

        return out


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, ac_fn='relu', norm_fn='bn'):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, M=None):
        message = self.attn(x, source, source, M=M)
        self.prob = self.attn.prob

        out = self.mlp(torch.cat([x, message], dim=1))
        return out


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, input_dim, feature_dim, layers, ac_fn='relu', norm_fn='bn'):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = MLP([input_dim] + layers + [feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores=None):
        if self.input_dim == 2:
            return self.encoder(kpts.transpose(1, 2))
        else:
            inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]  # [B, 2, N] + [B, 1, N]
            return self.encoder(torch.cat(inputs, dim=1))
