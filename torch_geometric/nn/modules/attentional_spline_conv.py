import torch
from torch.nn import Module, Parameter

from .utils.inits import uniform
from .utils.repr import repr
from .utils.repeat import repeat_to
from ..functional.attentional_spline_conv import attentional_spline_conv


class AttSplineConv(Module):
    def __init__(self,
                 in_features,
                 out_features,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(AttSplineConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        kernel_size = torch.LongTensor(repeat_to(kernel_size, dim))
        self.register_buffer('kernel_size', kernel_size)
        is_open_spline = torch.ByteTensor(repeat_to(is_open_spline, dim))
        self.register_buffer('is_open_spline', is_open_spline)
        self.degree = degree
        self.negative_slope = negative_slope
        self.dropout = dropout

        weight = torch.Tensor(kernel_size.prod(), in_features, out_features)
        self.weight = Parameter(weight)
        root_weight = torch.Tensor(in_features, out_features)
        self.root_weight = Parameter(root_weight)
        self.att_weight = Parameter(torch.Tensor(2 * out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0) * self.in_features
        uniform(size, self.weight)
        uniform(size, self.root_weight)
        uniform(size, self.att_weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, pseudo):
        dropout = self.dropout if self.training else 0
        return attentional_spline_conv(
            x, edge_index, pseudo, self.weight, self._buffers['kernel_size'],
            self._buffers['is_open_spline'], self.degree, self.att_weight,
            self.negative_slope, dropout, self.root_weight, self.bias)

    def __repr__(self):
        return repr(self, ['degree'])
