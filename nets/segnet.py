# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> segnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/01/2024 14:46
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.layers import MLP, KeypointEncoder, normalize_keypoints
from nets.layers import AttentionalPropagation


class SegGNN(nn.Module):
    def __init__(self, feature_dim: int, n_layers: int, ac_fn: str = 'relu', norm_fn: str = 'bn', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4, ac_fn=ac_fn, norm_fn=norm_fn)
            for _ in range(n_layers)
        ])

    def forward(self, desc):
        for i, layer in enumerate(self.layers):
            delta = layer(desc, desc)
            desc = desc + delta

        return desc


class SegNet(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'output_dim': 1024,
        'n_class': 512,
        'keypoint_encoder': [32, 64, 128, 256],
        'n_layers': 9,
        'ac_fn': 'relu',
        'norm_fn': 'in',
        'with_score': False,
        # 'with_global': False,
        'with_cls': False,
        'with_sc': False,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.with_cls = self.config['with_cls']
        self.with_sc = self.config['with_sc']

        self.n_layers = self.config['n_layers']
        self.gnn = SegGNN(
            feature_dim=self.config['descriptor_dim'],
            n_layers=self.config['n_layers'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
        )

        self.with_score = self.config['with_score']
        self.kenc = KeypointEncoder(
            input_dim=3 if self.with_score else 2,
            feature_dim=self.config['descriptor_dim'],
            layers=self.config['keypoint_encoder'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn']
        )

        self.seg = MLP(channels=[self.config['descriptor_dim'],
                                 self.config['output_dim'],
                                 self.config['n_class']],
                       ac_fn=self.config['ac_fn'],
                       norm_fn=self.config['norm_fn']
                       )

        if self.with_sc:
            self.sc = MLP(channels=[self.config['descriptor_dim'],
                                    self.config['output_dim'],
                                    3],
                          ac_fn=self.config['ac_fn'],
                          norm_fn=self.config['norm_fn']
                          )

    def preprocess(self, data):
        desc0 = data['descriptors']
        desc0 = desc0.transpose(1, 2)  # [B, D, N]

        if 'norm_keypoints' in data.keys():
            norm_kpts0 = data['norm_keypoints']
        elif 'image' in data.keys():
            kpts0 = data['keypoints']
            norm_kpts0 = normalize_keypoints(kpts0, data['image'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        if self.with_score:
            scores0 = data['scores']
        else:
            scores0 = None
        enc0 = self.kenc(norm_kpts0, scores0)

        return desc0, enc0

    def forward(self, data):
        desc, enc = self.preprocess(data=data)
        desc = desc + enc

        desc = self.gnn(desc)
        cls_output = self.seg(desc)  # [B, C, N]
        output = {
            'prediction': cls_output,
        }

        if self.with_sc:
            sc_output = self.sc(desc)
            output['sc'] = sc_output

        return output
