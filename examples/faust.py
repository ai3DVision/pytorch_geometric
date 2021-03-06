from __future__ import division, print_function

import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.datasets import FAUST
from torch_geometric.transform import CartesianAdj
from torch_geometric.utils import DataLoader
from torch_geometric.nn.modules import SplineConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FAUST')
train_dataset = FAUST(path, train=True, transform=CartesianAdj())
test_dataset = FAUST(path, train=False, transform=CartesianAdj())
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 6890)

    def forward(self, data):
        x, edge_index, pseudo = data.input, data.index, data.weight
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
target = torch.arange(0, model.fc2.weight.size(0)).long()
if torch.cuda.is_available():
    model, target = model.cuda(), target.cuda()

target = Variable(target)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 61:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in train_loader:
        data = data.cuda().to_variable(['input', 'weight'])
        optimizer.zero_grad()
        output = model(data)
        F.nll_loss(output, target).backward()
        optimizer.step()


def test(epoch):
    model.eval()
    correct = 0

    for i, data in enumerate(test_loader):
        data = data.cuda().to_variable(['input', 'weight'])
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).sum()

    print('Epoch', epoch, 'Test Accuracy:', correct / (20 * 6890))


for epoch in range(1, 101):
    train(epoch)
    test(epoch)
