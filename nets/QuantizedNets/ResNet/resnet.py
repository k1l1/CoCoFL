from nets.QuantizedNets.ResNet.training import Bottleneck
from nets.QuantizedNets.ResNet.backward import QBWBottleneck
from nets.QuantizedNets.ResNet.forward import QFWBottleneck
from nets.QuantizedNets.ResNet.backward import QBWBasicBlock
from nets.QuantizedNets.ResNet.training import BasicBlock
from nets.QuantizedNets.ResNet.forward import QFWBasicBlock
import random
import torch.nn as nn
import torch.nn.functional as F

import json

from nets.QuantizedNets.ResNet.training import InputLayer
from nets.QuantizedNets.utils.utils import filter_table

with open('nets/QuantizedNets/ResNet/tables/table__CoCoFL_arm_QResNet18.json', 'r') as fd:
    _g_table_qresnet18 = json.load(fd)
with open('nets/QuantizedNets/ResNet/tables/table__CoCoFL_x64_QResNet50.json', 'r') as fd:
    _g_table_qresnet50 = json.load(fd)


class QResNet(nn.Module):
    def __init__(self, trained_block, fw_block, bw_block, num_blocks, num_classes=10, freeze_idxs=[]):
        super(QResNet, self).__init__()

        self.in_planes = 64

        layer_idxs = [i for i in range(1 + 1 + sum(num_blocks))]
        assert set(freeze_idxs) <= set(layer_idxs), "Invalid layer idxs"

        self._trained_idxs = [idx for idx in layer_idxs if idx not in freeze_idxs]
        assert set(self._trained_idxs) == set([i for i in range(max(self._trained_idxs) + 1 - len(self._trained_idxs), max(self._trained_idxs) + 1)]), \
             "No continous block of trained layers"

        self._trained_block = trained_block
        self._fw_block = fw_block
        self._bw_block = bw_block

        self.input_layer = InputLayer()

        if 0 not in self._trained_idxs:
            for parameter in self.input_layer.parameters():
                parameter.requires_grad = False

        self._block_idx = 1

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*self._trained_block.expansion, num_classes)

        # freeze linear layer in case it does not get trained
        if (1 + sum(num_blocks)) not in self._trained_idxs:
            for parameter in self.linear.parameters():
                parameter.requires_grad = False

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for st in strides:
            # case fw layer (frozen)
            if self._block_idx < min(self._trained_idxs):
                transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
                layers.append(self._fw_block(self.in_planes, planes, st, is_transition=transition))

            # case bw+(fw) layer (also frozen)
            elif self._block_idx > max(self._trained_idxs):
                layers.append(self._bw_block(self.in_planes, planes, st))

            # trained layer
            elif self._block_idx in self._trained_idxs:
                layers.append(self._trained_block(self.in_planes, planes, st))
            else: raise ValueError

            self._block_idx += 1
            self.in_planes = planes * self._trained_block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x




class QResNet18(QResNet):
    def __init__(self, num_classes=10, freeze=[]):
        super(QResNet18, self).__init__(BasicBlock, QFWBasicBlock, QBWBasicBlock,
                                        [2, 2, 2, 2], num_classes=num_classes, freeze_idxs=freeze)

    @staticmethod
    def n_freezable_layers():
        return 1 + sum([2, 2, 2, 2]) + 1

    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qresnet18, QResNet18.n_freezable_layers())
        return random.choice(configs)


### RESNET50 ###


class QResNet50(QResNet):
    def __init__(self, num_classes=10, freeze=[]):
        super(QResNet50, self).__init__(Bottleneck, QFWBottleneck, QBWBottleneck,
                                        [3, 4, 6, 3], num_classes=num_classes, freeze_idxs=freeze)

    @staticmethod
    def n_freezable_layers():
        return 1 + sum([3, 4, 6, 3]) + 1

    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qresnet50, QResNet50.n_freezable_layers())
        return random.choice(configs)