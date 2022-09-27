import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['MovingLayerNorm1d', 'MovingLayerNorm2d',
           'MovingBatchNorm1d', 'MovingBatchNorm2d']


# -------------------------------------------------------------------------------------- #
class MovingLayerNormNd(nn.Module):   # not done
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(MovingLayerNormNd, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# -------------------------------------------------------------------------------------- #
class MovingBatchNormNd(nn.Module):
    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        if self.affine:
            self.bias.data.zero_()

    def forward(self, x, logpx=None):
        c = x.size(1)
        used_mean = self.running_mean.clone().detach()
        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).contiguous().view(c, -1)
            batch_mean = torch.mean(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= (1. - self.bn_lag ** (self.step[0] + 1))

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)
        y = x - used_mean
        if self.affine:
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y + bias
        if logpx is None:
            return y
        else:
            return y, logpx

    def inverse(self, y, logpy=None):
        used_mean = self.running_mean

        if self.affine:
            bias = self.bias.view(*self.shape).expand_as(y)
            y = y - bias

        used_mean = used_mean.view(*self.shape).expand_as(y)
        x = y + used_mean

        if logpy is None:
            return x
        else:
            return x, logpy

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
            ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__))


# -------------------------------------------------------------------------------------- #
class MovingLayerNorm1d(MovingLayerNormNd):
    @property
    def shape(self):
        return [1, -1]


class MovingLayerNorm2d(MovingLayerNormNd):
    @property
    def shape(self):
        return [1, -1, 1, 1]


class MovingBatchNorm1d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1]


class MovingBatchNorm2d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1, 1, 1]


#############################################################
if __name__ == '__main__':
    def test_MovingBatchNorm1d():
        MBN = MovingBatchNorm1d(num_features=64)
        input = torch.rand(64, 64)
        output = MBN(input)
        print('output moving batch norm shape:', output.shape)

    def test_MovingBatchNorm2d():
        MBN = MovingBatchNorm2d(num_features=64)
        input = torch.rand(64, 64, 64, 64)
        output = MBN(input)
        print('output moving batch norm shape:', output.shape)

    test_MovingBatchNorm1d()
    test_MovingBatchNorm2d()


