from models.layers.pooling import MAC, SPoC, GeM, NetVLADWrapper
import torch.nn as nn
import MinkowskiEngine as ME


class PoolingWrapper(nn.Module):
    def __init__(self, pool_method, in_dim, output_dim):
        super().__init__()

        self.pool_method = pool_method
        self.in_dim = in_dim
        self.output_dim = output_dim

        if pool_method == 'MAC':
            # Global max pooling
            assert in_dim == output_dim
            self.pooling = MAC(input_dim=in_dim)
        elif pool_method == 'SPoC':
            # Global average pooling
            assert in_dim == output_dim
            self.pooling = SPoC(input_dim=in_dim)
        elif pool_method == 'GeM':
            # Generalized mean pooling
            assert in_dim == output_dim
            self.pooling = GeM(input_dim=in_dim)
        elif self.pool_method == 'netvlad':
            # NetVLAD
            self.pooling = NetVLADWrapper(feature_size=in_dim, output_dim=output_dim, gating=False)
        elif self.pool_method == 'netvladgc':
            # NetVLAD with Gating Context
            self.pooling = NetVLADWrapper(feature_size=in_dim, output_dim=output_dim, gating=True)
        else:
            raise NotImplementedError('Unknown pooling method: {}'.format(pool_method))

    def forward(self, x: ME.SparseTensor):
        return self.pooling(x)
