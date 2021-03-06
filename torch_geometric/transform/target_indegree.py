from __future__ import division

import torch


class TargetIndegreeAdj(object):
    def __call__(self, data):
        _, col = data.index

        zero = torch.zeros(data.num_nodes, out=col.new().float())
        one = torch.ones(col.size(0), out=zero.new())

        degree = zero.scatter_add_(0, col, one)
        degree /= degree.max()
        degree = degree[col]

        if data.weight is None:
            data.weight = degree
        else:
            degree, weight = degree.unsqueeze(1), data.weight.unsqueeze(1)
            data.weight = torch.cat([degree, weight], dim=1)

        return data
