from __future__ import division, print_function

import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import Cora
from torch_geometric.nn.modules import GAT

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = Cora(osp.join(path, 'Cora'), normalize=True)
data = dataset[0].cuda().to_variable()
num_features, num_targets = data.input.size(1), data.target.data.max() + 1
train_mask = torch.arange(0, 20 * num_targets).long()
val_mask = torch.arange(train_mask.size(0), train_mask.size(0) + 500).long()
test_mask = torch.arange(data.num_nodes - 1000, data.num_nodes).long()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.att1 = GAT(num_features, 8, 8, dropout=0.6)
        self.att2 = GAT(64, num_targets, dropout=0.6)

    def forward(self):
        x = F.dropout(data.input, p=0.6, training=self.training)
        x = F.elu(self.att1(data.input, data.index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.att2(x, data.index)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    train_mask, val_mask = train_mask.cuda(), val_mask.cuda()
    test_mask, model = test_mask.cuda(), model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)


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
    model.att1.reset_parameters()
    model.att2.reset_parameters()

    old_val = 0
    cur_test = 0
    for _ in range(200):
        train()
        val = test(val_mask)
        if val > old_val:
            old_val = val
            cur_test = test(test_mask)

    acc.append(cur_test)
    print('Run:', run, 'Test Accuracy:', acc[-1])

print('Mean:', torch.Tensor(acc).mean(), 'Stddev:', torch.Tensor(acc).std())
