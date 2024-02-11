# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> adagml
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   11/02/2024 14:29
=================================================='''
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable
import time
import numpy as np

torch.backends.cudnn.deterministic = True

eps = 1e-8


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def dual_softmax(M, dustbin):
    M = torch.cat([M, dustbin.expand([M.shape[0], M.shape[1], 1])], dim=-1)
    M = torch.cat([M, dustbin.expand([M.shape[0], 1, M.shape[2]])], dim=-2)
    score = torch.log_softmax(M, dim=-1) + torch.log_softmax(M, dim=1)
    return torch.exp(score)


def sinkhorn(M, r, c, iteration):
    p = torch.softmax(M, dim=-1)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iteration):
        u = r / ((p * v.unsqueeze(-2)).sum(-1) + eps)
        v = c / ((p * u.unsqueeze(-1)).sum(-2) + eps)
    p = p * u.unsqueeze(-1) * v.unsqueeze(-2)
    return p


def sink_algorithm(M, dustbin, iteration):
    M = torch.cat([M, dustbin.expand([M.shape[0], M.shape[1], 1])], dim=-1)
    M = torch.cat([M, dustbin.expand([M.shape[0], 1, M.shape[2]])], dim=-2)
    r = torch.ones([M.shape[0], M.shape[1] - 1], device='cuda')
    r = torch.cat([r, torch.ones([M.shape[0], 1], device='cuda') * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1], device='cuda')
    c = torch.cat([c, torch.ones([M.shape[0], 1], device='cuda') * M.shape[2]], dim=-1)
    p = sinkhorn(M, r, c, iteration)
    return p


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(
        freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None,
                 gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ encode position vector """
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(128, 256),
        )

    def forward(self, kpts, scores):
        inputs = [kpts, scores.unsqueeze(2)]  # [B, N, 2] + [B, N, 1]
        return self.encoder(torch.cat(inputs, dim=-1))


class PoolingLayer(nn.Module):
    def __init__(self, hidden_dim: int, score_dim: int = 2):
        super().__init__()

        self.score_enc = nn.Sequential(
            nn.Linear(score_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, score):
        score_ = self.score_enc(score)
        x_ = self.proj(x)
        confidence = self.predict(torch.cat([x_, score_], -1))
        confidence = torch.sigmoid(confidence)

        return confidence


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        s = q.shape[-1] ** -0.5
        attn = F.softmax(torch.einsum('...id,...jd->...ij', q, k) * s, -1)
        return torch.einsum('...ij,...jd->...id', attn, v), torch.mean(torch.mean(attn, dim=1), dim=1)


class SelfMultiHeadAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        assert feat_dim % num_heads == 0
        self.head_dim = feat_dim // num_heads
        self.qkv = nn.Linear(feat_dim, hidden_dim * 3)
        self.attn = Attention()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim)
        )

    def forward_(self, x, encoding=None):
        qkv = self.qkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if encoding is not None:
            q = apply_cached_rotary_emb(encoding, q)
            k = apply_cached_rotary_emb(encoding, k)
        attn, attn_score = self.attn(q, k, v)
        message = self.proj(attn.transpose(1, 2).flatten(start_dim=-2))
        return x + self.mlp(torch.cat([x, message], -1)), attn_score

    def forward(self, x0, x1, encoding0=None, encoding1=None):
        x0_, att_score00 = self.forward_(x=x0, encoding=encoding0)
        x1_, att_score11 = self.forward_(x=x1, encoding=encoding1)
        return x0_, x1_, att_score00, att_score11


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        dim_head = hidden_dim // num_heads
        self.scale = dim_head ** -0.5
        self.to_qk = nn.Linear(feat_dim, hidden_dim)
        self.to_v = nn.Linear(feat_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(self, x0, x1):
        qk0 = self.to_qk(x0)
        qk1 = self.to_qk(x1)
        v0 = self.to_v(x0)
        v1 = self.to_v(x1)

        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.num_heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1))

        qk0, qk1 = qk0 * self.scale ** 0.5, qk1 * self.scale ** 0.5
        sim = torch.einsum('b h i d, b h j d -> b h i j', qk0, qk1)
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        m0 = torch.einsum('bhij, bhjd -> bhid', attn01, v1)
        m1 = torch.einsum('bhji, bhjd -> bhid', attn10.transpose(-2, -1), v0)

        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2),
                           m0, m1)
        m0, m1 = self.map_(self.proj, m0, m1)
        x0 = x0 + self.mlp(torch.cat([x0, m0], -1))
        x1 = x1 + self.mlp(torch.cat([x1, m1], -1))
        return x0, x1, torch.mean(torch.mean(attn10, dim=1), dim=1), torch.mean(torch.mean(attn01, dim=1), dim=1)


class AdaGML(nn.Module):
    default_config = {
        'descriptor_dim': 128,
        'hidden_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,  # [self, cross, self, cross, ...] 9 in total
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'with_pose': True,
        'n_layers': 9,
        'n_min_tokens': 256,
        'with_sinkhorn': True,
        'min_confidence': 0.9,

        'classification_background_weight': 0.05,
        'pretrained': True,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.n_layers = self.config['n_layers']
        self.first_layer_pooling = 0
        self.n_min_tokens = self.config['n_min_tokens']
        self.min_confidence = self.config['min_confidence']
        self.classification_background_weight = self.config['classification_background_weight']

        self.with_sinkhorn = self.config['with_sinkhorn']
        self.match_threshold = self.config['match_threshold']
        self.sinkhorn_iterations = self.config['sinkhorn_iterations']

        self.input_proj = nn.Linear(self.config['descriptor_dim'], self.config['hidden_dim'])

        self.self_attn = nn.ModuleList(
            [SelfMultiHeadAttention(feat_dim=self.config['hidden_dim'],
                                    hidden_dim=self.config['hidden_dim'],
                                    num_heads=4) for _ in range(self.n_layers)]
        )
        self.cross_attn = nn.ModuleList(
            [CrossMultiHeadAttention(feat_dim=self.config['hidden_dim'],
                                     hidden_dim=self.config['hidden_dim'],
                                     num_heads=4) for _ in range(self.n_layers)]
        )

        head_dim = self.config['hidden_dim'] // 4
        self.poseenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)
        self.out_proj = nn.ModuleList(
            [nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']) for _ in range(self.n_layers)]
        )

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.pooling = nn.ModuleList(
            [PoolingLayer(score_dim=2, hidden_dim=self.config['hidden_dim']) for _ in range(self.n_layers)]
        )
        # self.pretrained = config['pretrained']
        # if self.pretrained:
        #     bin_score.requires_grad = False
        #     for m in [self.input_proj, self.out_proj, self.poseenc, self.self_attn, self.cross_attn]:
        #         for p in m.parameters():
        #             p.requires_grad = False

    def forward(self, data, mode=0):
        if not self.training:
            if mode == 0:
                return self.produce_matches(data=data)
            else:
                return self.run(data=data)
        return self.forward_train(data=data)

    def forward_train(self, data: dict, p=0.2, **kwargs):
        pass

    def produce_matches(self, data: dict, p: float = 0.2, **kwargs):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

        # Keypoint normalization.
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        elif 'image_shape0' in data.keys() and 'image_shape1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image_shape0'])
            norm_kpts1 = normalize_keypoints(kpts1, data['image_shape1'])
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        desc0 = desc0.detach()  # [B, N, D]
        desc1 = desc1.detach()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        enc0 = self.poseenc(norm_kpts0)
        enc1 = self.poseenc(norm_kpts1)

        nI = self.config['n_layers']
        nB = desc0.shape[0]
        m = desc0.shape[1]
        n = desc1.shape[1]
        dev = desc0.device

        ind0 = torch.arange(0, m, device=dev)[None]
        ind1 = torch.arange(0, n, device=dev)[None]

        do_pooling = True

        for ni in range(nI):
            desc0, desc1, att_score00, att_score11 = self.self_attn[ni](desc0, desc1, enc0, enc1)
            desc0, desc1, att_score01, att_score10 = self.cross_attn[ni](desc0, desc1)

            att_score0 = torch.cat([att_score00.unsqueeze(-1), att_score01.unsqueeze(-1)], dim=-1)
            att_score1 = torch.cat([att_score11.unsqueeze(-1), att_score10.unsqueeze(-1)], dim=-1)

            conf0 = self.pooling[ni](desc0, att_score0).squeeze(-1)
            conf1 = self.pooling[ni](desc1, att_score1).squeeze(-1)

            if do_pooling and ni >= 1:
                if desc0.shape[1] >= self.n_min_tokens:
                    mask0 = conf0 > self.confidence_threshold(layer_index=ni)
                    ind0 = ind0[mask0][None]
                    desc0 = desc0[mask0][None]
                    enc0 = enc0[:, :, mask0][:, None]

                if desc1.shape[1] >= self.n_min_tokens:
                    mask1 = conf1 > self.confidence_threshold(layer_index=ni)
                    ind1 = ind1[mask1][None]
                    desc1 = desc1[mask1][None]
                    enc1 = enc1[:, :, mask1][:, None]

                # print('pooling: ', ni, desc0.shape, desc1.shape)
                # print('ni: {:d}: pooling: {:.4f}'.format(ni, time.time() - t_start))
                # t_start = time.time()
                if self.check_if_stop(confidences0=conf0, confidences1=conf1, layer_index=ni, num_points=m + n):
                    # print('ni:{:d}: checking: {:.4f}'.format(ni, time.time() - t_start))
                    break

        if ni == nI: ni = nI - 1
        d = desc0.shape[-1]
        mdesc0 = self.out_proj[ni](desc0) / d ** .25
        mdesc1 = self.out_proj[ni](desc1) / d ** .25

        dist = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score, p=p)
        valid = indices0 > -1
        m_indices0 = torch.where(valid)[1]
        m_indices1 = indices0[valid]

        mind0 = ind0[0, m_indices0]
        mind1 = ind1[0, m_indices1]

        indices0_full = torch.full((nB, m), -1, device=dev, dtype=indices0.dtype)
        indices0_full[:, mind0] = mind1

        mscores0_full = torch.zeros((nB, m), device=dev)
        mscores0_full[:, ind0] = mscores0

        indices0 = indices0_full
        mscores0 = mscores0_full

        output = {
            'matches0': indices0,  # use -1 for invalid match
            # 'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
        }

        return output

    def run(self, data, p=0.2):
        desc0 = data['desc1']
        # print('desc0: ', torch.sum(desc0 ** 2, dim=-1))
        # desc0 = torch.nn.functional.normalize(desc0, dim=-1)
        desc0 = desc0.detach()

        desc1 = data['desc2']
        # desc1 = torch.nn.functional.normalize(desc1, dim=-1)
        desc1 = desc1.detach()

        kpts0 = data['x1'][:, :, :2]
        kpts1 = data['x2'][:, :, :2]
        # kpts0 = normalize_keypoints(kpts=kpts0, image_shape=data['image_shape1'])
        # kpts1 = normalize_keypoints(kpts=kpts1, image_shape=data['image_shape2'])
        scores0 = data['x1'][:, :, -1]
        scores1 = data['x2'][:, :, -1]

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        enc0 = self.poseenc(kpts0)
        enc1 = self.poseenc(kpts1)

        nB = desc0.shape[0]
        nI = self.n_layers
        m, n = desc0.shape[1], desc1.shape[1]
        dev = desc0.device
        ind0 = torch.arange(0, m, device=dev)[None]
        ind1 = torch.arange(0, n, device=dev)[None]
        do_pooling = True

        for ni in range(nI):
            desc0, desc1, att_score00, att_score11 = self.self_attn[ni](desc0, desc1, enc0, enc1)
            desc0, desc1, att_score01, att_score10 = self.cross_attn[ni](desc0, desc1)

            att_score0 = torch.cat([att_score00.unsqueeze(-1), att_score01.unsqueeze(-1)], dim=-1)
            att_score1 = torch.cat([att_score11.unsqueeze(-1), att_score10.unsqueeze(-1)], dim=-1)

            conf0 = self.pooling[ni](desc0, att_score0).squeeze(-1)
            conf1 = self.pooling[ni](desc1, att_score1).squeeze(-1)

            if do_pooling and ni >= 1:
                if desc0.shape[1] >= self.n_min_tokens:
                    mask0 = conf0 > self.confidence_threshold(layer_index=ni)
                    ind0 = ind0[mask0][None]
                    desc0 = desc0[mask0][None]
                    enc0 = enc0[:, :, mask0][:, None]

                if desc1.shape[1] >= self.n_min_tokens:
                    mask1 = conf1 > self.confidence_threshold(layer_index=ni)
                    ind1 = ind1[mask1][None]
                    desc1 = desc1[mask1][None]
                    enc1 = enc1[:, :, mask1][:, None]
                if desc0.shape[1] <= 5 or desc1.shape[1] <= 5:
                    return {
                        'index0': torch.zeros(size=(1,), device=desc0.device).long(),
                        'index1': torch.zeros(size=(1,), device=desc1.device).long(),
                    }

                if self.check_if_stop(confidences0=conf0, confidences1=conf1, layer_index=ni,
                                      num_points=m + n):
                    break

        if ni == nI: ni = -1
        d = desc0.shape[-1]
        mdesc0 = self.out_proj[ni](desc0) / d ** .25
        mdesc1 = self.out_proj[ni](desc1) / d ** .25

        dist = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score, p=p)
        valid = indices0 > -1
        m_indices0 = torch.where(valid)[1]
        m_indices1 = indices0[valid]

        mind0 = ind0[0, m_indices0]
        mind1 = ind1[0, m_indices1]

        output = {
            # 'p': score,
            'index0': mind0,
            'index1': mind1,
        }

        return output

    def compute_score(self, dist, dustbin, iteration):
        if self.with_sinkhorn:
            score = sink_algorithm(M=dist, dustbin=dustbin,
                                   iteration=iteration)  # [nI * nB, N, M]
        else:
            score = dual_softmax(M=dist, dustbin=dustbin)
        return score

    def compute_matches(self, scores, p=0.2):
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores0 = torch.where(mutual0, max0.values, zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid0 = mutual0 & (mscores0 > p)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0, indices1, mscores0, mscores1

    def confidence_threshold(self, layer_index: int):
        """scaled confidence threshold"""
        # threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        threshold = 0.5 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return np.clip(threshold, 0, 1)

    def check_if_stop(self,
                      confidences0: torch.Tensor,
                      confidences1: torch.Tensor,
                      layer_index: int, num_points: int) -> torch.Tensor:
        """ evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_threshold(layer_index)
        pos = 1.0 - (confidences < threshold).float().sum() / num_points
        # print('check_stop: ', pos)
        return pos > 0.95

    def stop_iteration(self, m_last, n_last, m_current, n_current, confidence=0.975):
        prob = (m_current + n_current) / (m_last + n_last)
        # print('prob: ', prob)
        return prob > confidence
