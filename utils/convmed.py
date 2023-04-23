"""Architecture implementation is based on the implementation found in:
https://raw.githubusercontent.com/eth-sri/colt/master/code/networks.py
"""

import torch
import torch.nn as nn
from functools import reduce


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, dim=None):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.dim = dim
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward_concrete(self, x):
        return self.conv(x)

    def forward_abstract(self, x):
        return x.conv2d(self.conv.weight, self.conv.bias, self.stride, self.conv.padding, self.dilation, self.conv.groups)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            ret = self.forward_concrete(x)
        else:
            ret = self.forward_abstract(x)
        return ret


class Sequential(nn.Module):

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward_until(self, i, x):
        for layer in self.layers[:i+1]:
            x = layer(x)
        return x

    def forward_from(self, i, x):
        for layer in self.layers[i+1:]:
            x = layer(x)
        return x

    def total_abs_l1(self, x):
        ret = 0
        for layer in self.layers:
            x = layer(x)
            ret += x.l1()
        return ret

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward(self, x, init_lambda=False, skip_norm=False):
        for layer in self.layers:
            if isinstance(layer, Normalization) and skip_norm:
                continue
            if isinstance(layer, ReLU):
                x = layer(x, init_lambda)
            else:
                x = layer(x)
        return x
        
    
class ReLU(nn.Module):

    def __init__(self, dims=None):
        super(ReLU, self).__init__()
        self.dims = dims
        self.deepz_lambda = nn.Parameter(torch.ones(dims))
        self.bounds = None

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def forward(self, x, init_lambda=False):
        # if isinstance(x, HybridZonotope):
        #     return x.relu(self.deepz_lambda, self.bounds, init_lambda)
        return x.relu()


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view((x.size()[0], -1))


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.linear(x)
        else:
            return x.linear(self.linear.weight, self.linear.bias)


class Normalization(nn.Module):

    def __init__(self, mean, sigma):
        super(Normalization, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / self.sigma
        ret = x.normalize(self.mean, self.sigma)
        return ret


def get_mean_sigma(device, dataset):
    if dataset == 'cifar10':
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))
    else:
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
    return mean.to(device), sigma.to(device)

class SeqNet(nn.Module):

    def __init__(self):
        super(SeqNet, self).__init__()
        self.is_double = False
        self.skip_norm = False

    def forward(self, x, init_lambda=False):
        if isinstance(x, torch.Tensor) and self.is_double:
            x = x.to(dtype=torch.float64)
        x = self.blocks(x, init_lambda, skip_norm=self.skip_norm)
        return x

    def reset_bounds(self):
        for block in self.blocks:
            block.bounds = None

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x


class FFNN(SeqNet):

    def __init__(self, device, dataset, sizes, n_class=10, input_size=32, input_channel=3):
        super(FFNN, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [Flatten(), Linear(input_size*input_size*input_channel, sizes[0]), ReLU(sizes[0])]
        for i in range(1, len(sizes)):
            layers += [
                Linear(sizes[i-1], sizes[i]),
                ReLU(sizes[i]),
            ]
        layers += [Linear(sizes[-1], n_class)]
        self.blocks = Sequential(*layers)


class ConvMed(SeqNet):

    def __init__(self, device='cuda', dataset='cifar10', n_class=10, input_size=32, input_channel=3, width1=2, width2=4, linear_size=250):
        super(ConvMed, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 5, stride=2, padding=2, dim=input_size),
            ReLU((16*width1, input_size//2, input_size//2)),
            Conv2d(16*width1, 32*width2, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((32*width2, input_size//4, input_size//4)),
            Flatten(),
            Linear(32*width2*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)

        
class ConvMedBig(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, width3=1, linear_size=100):
        super(ConvMedBig, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 3, stride=1, padding=1, dim=input_size),
            ReLU((16*width1, input_size, input_size)),
            Conv2d(16*width1, 16*width2, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((16*width2, input_size//2, input_size//2)),
            Conv2d(16*width2, 32*width3, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((32*width3, input_size//4, input_size//4)),
            Flatten(),
            Linear(32*width3*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)
