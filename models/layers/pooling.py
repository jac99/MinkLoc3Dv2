# Pooling methods code based on: https://github.com/filipradenovic/cnnimageretrieval-pytorch

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from models.layers.netvlad import NetVLADLoupe


class MAC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class SPoC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class GeM(nn.Module):
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        #temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1./self.p)    # Return (batch_size, n_features) tensor


class NetVLADWrapper(nn.Module):
    def __init__(self, feature_size, output_dim, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.net_vlad = NetVLADLoupe(feature_size=feature_size, cluster_size=64, output_dim=output_dim, gating=gating,
                                     add_batch_norm=True)

    def forward(self, x: ME.SparseTensor):
        # x is (batch_size, C, H, W)
        assert x.F.shape[1] == self.feature_size
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        batch_size = len(features)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros

        x = self.net_vlad(features)
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.output_dim
        return x    # Return (batch_size, output_dim) tensor
