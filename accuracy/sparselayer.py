import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np

from compress import blocksparse, blocksparse_cpu

__all__ = ['BlocksparseConv', 'BlocksparseLinear']

class BlocksparseConv(nn.Module):
    def __init__(self, conv2d, block_sizes, pruning_rate, shuffle=True):
        super(BlocksparseConv, self).__init__()

        self.block_sizes = block_sizes
        self.pruning_rate = pruning_rate

        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size
        self.stride = conv2d.stride
        self.padding = conv2d.padding
        self.dilation = conv2d.dilation
        self.transposed = conv2d.transposed
        self.output_padding = conv2d.output_padding
        self.groups = conv2d.groups
        
        self.weight = Parameter(conv2d.weight.data)
        if conv2d.bias is not None:
            self.bias = Parameter(conv2d.bias.data)
        else:
            self.register_parameter('bias', None)

        W = self.weight.data
        orders, mask = blocksparse(W, block_sizes, pruning_rate, shuffle)
        mask = Variable(mask)
        self.register_buffer('mask', mask)
        print("Sparsity: %f, pruning_rate: %f" % (1 - self.mask.sum() / len(self.mask.view(-1)), pruning_rate))

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BlocksparseLinear(nn.Module):
    def __init__(self, linear, block_sizes, pruning_rate, shuffle=True):
        super(BlocksparseLinear, self).__init__()
        
        self.block_sizes = block_sizes
        self.pruning_rate = pruning_rate

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data)
        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data)
        else:
            self.register_parameter('bias', None)

        W = self.weight.data
        orders, mask = blocksparse(W, block_sizes, pruning_rate, shuffle)
        mask = Variable(mask)
        self.register_buffer('mask', mask)
        print("Sparsity: %f, pruning_rate: %f" % (1 - self.mask.sum() / len(self.mask.view(-1)), pruning_rate))

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        return F.linear(x, self.weight, self.bias)
