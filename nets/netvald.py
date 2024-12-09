import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torchvision.models as models


# code from https://github.com/Nanne/pytorch-NetVlad/blob/master/netvlad.py

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def load_netvlad(weight_path: str, encoder_name: str = 'vgg16', pretrained=True, num_clusters=64, ):
    if encoder_name == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False
    elif encoder_name == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False
    else:
        raise 'Please specify an encoder name either vgg16 or alexnet.'
    net_vlad = NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=False)
    net_vlad.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
    return net_vlad
