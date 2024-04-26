# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> adagml
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   11/02/2024 14:34
=================================================='''
import torch
from localization.base_model import BaseModel
from nets.adagml import AdaGML as GMatcher


class AdaGML(BaseModel):
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
        'weight_path': None,
    }

    required_inputs = [
        'image0', 'keypoints0', 'scores0', 'descriptors0',
        'image1', 'keypoints1', 'scores1', 'descriptors1',
    ]

    def _init(self, conf):
        self.net = GMatcher(config=conf).eval()
        state_dict = torch.load(conf['weight_path'], map_location='cpu')['model']
        self.net.load_state_dict(state_dict, strict=True)

    def _forward(self, data):
        with torch.no_grad():
            return self.net(data)
