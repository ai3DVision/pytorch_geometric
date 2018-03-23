from __future__ import division, print_function

import os.path as osp
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cora  # noqa
from torch_geometric.transform import TargetIndegreeAdj  # noqa
from torch_geometric.nn.modules import AttSplineConv  # noqa

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = Cora(osp.join(path, 'Cora'), transform=TargetIndegreeAdj())
data = dataset[0].cuda().to_variable()
train_mask = torch.arange(0, data.num_nodes - 1000).long()
test_mask = torch.arange(data.num_nodes - 500, data.num_nodes).long()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = AttSplineConv(1433, 8, dim=1, dropout=0.6, kernel_size=2)
        self.conv2 = AttSplineConv(8, 7, dim=1, dropout=0.6, kernel_size=2)

    def forward(self):
        x, edge_index, pseudo = data.input, data.index, data.weight
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, pseudo)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    train_mask, test_mask = train_mask.cuda(), test_mask.cuda()
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_mask], data.target[train_mask]).backward()
    optimizer.step()


def test(mask):
    model.eval()
    pred = model()[mask].data.max(1)[1]
    return pred.eq(data.target.data[mask]).sum() / mask.size(0)


acc = []
for run in range(1, 101):
    model.conv1.reset_parameters()
    model.conv2.reset_parameters()

    for _ in range(500):
        train()

    acc.append(test(test_mask))
    print('Run:', run, 'Test Accuracy:', acc[-1])

print('Mean:', torch.Tensor(acc).mean(), 'Stddev:', torch.Tensor(acc).std())
