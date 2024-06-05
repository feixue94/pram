# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> gm
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/02/2024 10:47
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.layers import KeypointEncoder, AttentionalPropagation
from nets.utils import normalize_keypoints, arange_like

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
    r = torch.ones([M.shape[0], M.shape[1] - 1], device='cuda')
    r = torch.cat([r, torch.ones([M.shape[0], 1], device='cuda') * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1], device='cuda')
    c = torch.cat([c, torch.ones([M.shape[0], 1], device='cuda') * M.shape[2]], dim=-1)
    p = sinkhorn(M, r, c, iteration)
    return p


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, hidden_dim: int = 256, ac_fn: str = 'relu',
                 norm_fn: str = 'bn'):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim=feature_dim, num_heads=4, hidden_dim=hidden_dim, ac_fn=ac_fn,
                                   norm_fn=norm_fn)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        # desc0s = []
        # desc1s = []

        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            delta0 = layer(desc0, src0)
            # prob0 = layer.attn.prob
            delta1 = layer(desc1, src1)
            # prob1 = layer.attn.prob
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

            # if name == 'cross':
            #     desc0s.append(desc0)
            #     desc1s.append(desc1)
        return [desc0], [desc1]

    def predict(self, desc0, desc1, n_it=-1):
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            delta0 = layer(desc0, src0)
            # prob0 = layer.attn.prob
            delta1 = layer(desc1, src1)
            # prob1 = layer.attn.prob
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

            if name == 'cross' and i == n_it:
                break
        return [desc0], [desc1]


class GM(nn.Module):
    default_config = {
        'descriptor_dim': 128,
        'hidden_dim': 256,
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
        'weight_path': None,
    }

    required_inputs = [
        'image0', 'keypoints0', 'scores0', 'descriptors0',
        'image1', 'keypoints1', 'scores1', 'descriptors1',
    ]

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        print('gm: ', self.config)

        self.n_layers = self.config['n_layers']

        self.with_sinkhorn = self.config['with_sinkhorn']
        self.match_threshold = self.config['match_threshold']

        self.sinkhorn_iterations = self.config['sinkhorn_iterations']
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'] if self.config['descriptor_dim'] > 0 else 128,
            self.config['keypoint_encoder'],
            layers=self.config['keypoint_encoder'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'])
        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'] if self.config['descriptor_dim'] > 0 else 128,
            hidden_dim=self.config['hidden_dim'],
            layer_names=self.config['GNN_layers'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
        )

        self.final_proj = nn.ModuleList([nn.Conv1d(
            self.config['descriptor_dim'] if self.config['descriptor_dim'] > 0 else 128,
            self.config['descriptor_dim'] if self.config['descriptor_dim'] > 0 else 128,
            kernel_size=1, bias=True) for _ in range(self.n_layers)])

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.match_net = None  # GraphLoss(config=self.config)

        self.self_prob0 = None
        self.self_prob1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

        self.desc_compressor = None

    def forward_train(self, data):
        pass

    def produce_matches(self, data, p=0.2, n_it=-1, **kwargs):
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

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

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)

        if self.config['descriptor_dim'] > 0:
            desc0, desc1 = data['descriptors0'], data['descriptors1']
            desc0 = desc0.transpose(0, 2, 1)  # [B, N, D ] -> [B, D, N]
            desc1 = desc1.transpose(0, 2, 1)  # [B, N, D ] -> [B, D, N]
            with torch.no_grad():
                if desc0.shape[1] != self.config['descriptor_dim']:
                    desc0 = self.desc_compressor(desc0)
                if desc1.shape[1] != self.config['descriptor_dim']:
                    desc1 = self.desc_compressor(desc1)
            desc0 = desc0 + enc0
            desc1 = desc1 + enc1
        else:
            desc0 = enc0
            desc1 = enc1

        desc0s, desc1s = self.gnn.predict(desc0, desc1, n_it=n_it)

        mdescs0 = self.final_proj[n_it](desc0s[-1])
        mdescs1 = self.final_proj[n_it](desc1s[-1])
        dist = torch.einsum('bdn,bdm->bnm', mdescs0, mdescs1)
        if self.config['descriptor_dim'] > 0:
            dist = dist / self.config['descriptor_dim'] ** .5
        else:
            dist = dist / 128 ** .5
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score, p=p)

        output = {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

        return output

    def forward(self, data, mode=0):
        if not self.training:
            return self.produce_matches(data=data, n_it=-1)
        return self.forward_train(data=data)

    def encode_keypoint(self, norm_kpts0, norm_kpts1, scores0, scores1):
        return self.kenc(norm_kpts0, scores0), self.kenc(norm_kpts1, scores1)

    def compute_distance(self, desc0, desc1, layer_id=-1):
        mdesc0 = self.final_proj[layer_id](desc0)
        mdesc1 = self.final_proj[layer_id](desc1)
        dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        dist = dist / self.config['descriptor_dim'] ** .5
        return dist

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
