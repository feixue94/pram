# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> segnetvit
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:52
=================================================='''

import torch
from torch import nn
import torch.nn.functional as F
from nets.utils import normalize_keypoints


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
            nn.Linear(2, 32),
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

    def forward(self, kpts, scores=None):
        if scores is not None:
            inputs = [kpts, scores.unsqueeze(2)]  # [B, N, 2] + [B, N, 1]
            return self.encoder(torch.cat(inputs, dim=-1))
        else:
            return self.encoder(kpts)


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

    def forward(self, x, encoding=None):
        qkv = self.qkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if encoding is not None:
            q = apply_cached_rotary_emb(encoding, q)
            k = apply_cached_rotary_emb(encoding, k)
        attn = self.attn(q, k, v)
        message = self.proj(attn.transpose(1, 2).flatten(start_dim=-2))
        return x + self.mlp(torch.cat([x, message], -1))


class SegGNNViT(nn.Module):
    def __init__(self, feature_dim: int, n_layers: int, hidden_dim: int = 256, num_heads: int = 4, **kwargs):
        super(SegGNNViT, self).__init__()
        self.layers = nn.ModuleList([
            SelfMultiHeadAttention(feat_dim=feature_dim, hidden_dim=hidden_dim, num_heads=num_heads)
            for _ in range(n_layers)
        ])

    def forward(self, desc, encoding=None):
        for i, layer in enumerate(self.layers):
            desc = layer(desc, encoding)
            # desc = desc + delta // should be removed as this is already done in self-attention
        return desc


class SegNetViT(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'output_dim': 1024,
        'n_class': 512,
        'keypoint_encoder': [32, 64, 128, 256],
        'n_layers': 9,
        'num_heads': 4,
        'hidden_dim': 256,
        'ac_fn': 'relu',
        'norm_fn': 'in',
        'with_score': False,
        'with_global': False,
        'with_cls': False,
        'with_sc': False,
    }

    def __init__(self, config={}):
        super(SegNetViT, self).__init__()
        self.config = {**self.default_config, **config}
        self.with_cls = self.config['with_cls']
        self.with_sc = self.config['with_sc']

        self.n_layers = self.config['n_layers']
        self.gnn = SegGNNViT(
            feature_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers'],
            hidden_dim=self.config['hidden_dim'],
            num_heads=self.config['num_heads'],
        )

        self.with_score = self.config['with_score']
        self.kenc = LearnableFourierPositionalEncoding(2, self.config['hidden_dim'] // self.config['num_heads'],
                                                       self.config['hidden_dim'] // self.config['num_heads'])

        self.input_proj = nn.Linear(in_features=self.config['descriptor_dim'],
                                    out_features=self.config['hidden_dim'])
        self.seg = nn.Sequential(
            nn.Linear(in_features=self.config['hidden_dim'], out_features=self.config['output_dim']),
            nn.LayerNorm(self.config['output_dim'], elementwise_affine=True),
            nn.GELU(),
            nn.Linear(self.config['output_dim'], self.config['n_class'])
        )

        if self.with_sc:
            self.sc = nn.Sequential(
                nn.Linear(in_features=config['hidden_dim'], out_features=self.config['output_dim']),
                nn.LayerNorm(self.config['output_dim'], elementwise_affine=True),
                nn.GELU(),
                nn.Linear(self.config['output_dim'], 3)
            )

    def preprocess(self, data):
        desc0 = data['descriptors']
        if 'norm_keypoints' in data.keys():
            norm_kpts0 = data['norm_keypoints']
        elif 'image' in data.keys():
            kpts0 = data['keypoints']
            norm_kpts0 = normalize_keypoints(kpts0, data['image'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        enc0 = self.kenc(norm_kpts0)

        return desc0, enc0

    def forward(self, data):
        desc, enc = self.preprocess(data=data)
        desc = self.input_proj(desc)

        desc = self.gnn(desc, enc)
        cls_output = self.seg(desc)  # [B, N, C]

        output = {
            'prediction': cls_output,
        }

        if self.with_sc:
            sc_output = self.sc(desc)
            output['sc'] = sc_output

        return output
