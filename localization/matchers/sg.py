# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram-dev -> sg
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   06/12/2024 15:04
=================================================='''
import torch
from localization.base_model import BaseModel
from nets.superglue import SuperGlue


class SG(BaseModel):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'weight_path': '/scratches/flyer_2/fx221/Research/Code/third_weights/superglue_outdoor.pth',
    }

    def _init(self, conf):
        self.conf = {**conf, **self.default_conf}
        self.net = SuperGlue(config=self.conf).eval()
        state_dict = torch.load(self.conf['weight_path'], map_location='cpu')
        self.net.load_state_dict(state_dict, strict=True)

    def _forward(self, data):
        with torch.no_grad():
            return self.net(data)
