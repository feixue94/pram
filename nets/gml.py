# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> gml
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 10:56
=================================================='''
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable
from nets.utils import arange_like, normalize_keypoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True

eps = 1e-8


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
    r = torch.ones([M.shape[0], M.shape[1] - 1], device=device)
    r = torch.cat([r, torch.ones([M.shape[0], 1], device=device) * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1], device=device)
    c = torch.cat([c, torch.ones([M.shape[0], 1], device=device) * M.shape[2]], dim=-1)
    p = sinkhorn(M, r, c, iteration)
    return p


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


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        s = q.shape[-1] ** -0.5
        attn = F.softmax(torch.einsum('...id,...jd->...ij', q, k) * s, -1)
        return torch.einsum('...ij,...jd->...id', attn, v)


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
        attn = self.attn(q, k, v)
        message = self.proj(attn.transpose(1, 2).flatten(start_dim=-2))
        return x + self.mlp(torch.cat([x, message], -1))

    def forward(self, x0, x1, encoding0=None, encoding1=None):
        return self.forward_(x0, encoding0), self.forward_(x1, encoding1)


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
        return x0, x1


class GML(nn.Module):
    '''
    the architecture of lightglue, but trained with imp
    '''
    default_config = {
        'descriptor_dim': 128,
        'hidden_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,  # [self, cross, self, cross, ...] 9 in total
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'with_pose': False,
        'n_layers': 9,
        'n_min_tokens': 256,
        'with_sinkhorn': True,

        'ac_fn': 'relu',
        'norm_fn': 'bn',

    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.n_layers = self.config['n_layers']

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

    def forward(self, data, mode=0):
        if not self.training:
            return self.produce_matches(data=data)
        return self.forward_train(data=data)

    def forward_train(self, data: dict, p=0.2, **kwargs):
        pass

    def produce_matches(self, data: dict, p=0.2, **kwargs):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        # Keypoint normalization.
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape).float()
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape).float()
        elif 'image_shape0' in data.keys() and 'image_shape1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image_shape0']).float()
            norm_kpts1 = normalize_keypoints(kpts1, data['image_shape1']).float()
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        enc0 = self.poseenc(norm_kpts0)
        enc1 = self.poseenc(norm_kpts1)

        nI = self.n_layers
        # nI = 5

        for i in range(nI):
            desc0, desc1 = self.self_attn[i](desc0, desc1, enc0, enc1)
            desc0, desc1 = self.cross_attn[i](desc0, desc1)

        d = desc0.shape[-1]
        mdesc0 = self.out_proj[nI - 1](desc0) / d ** .25
        mdesc1 = self.out_proj[nI - 1](desc1) / d ** .25

        dist = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)

        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score, p=p)

        output = {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
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
