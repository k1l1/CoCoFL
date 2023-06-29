from nets.QuantizedNets.utils.utils import filter_table
from nets.QuantizedNets.DenseNet.backwards import QBWBottleneck, QBWTransition
from nets.QuantizedNets.DenseNet.forward import QFWBottleneck, QFWTransition
from nets.QuantizedNets.DenseNet.training import Bottleneck, Transition
import torch.nn as nn
import math
import random
import json

with open('nets/QuantizedNets/DenseNet/tables/table__CoCoFL_arm_QDenseNet40.json', 'r') as fd:
    _g_table_qdensenet40 = json.load(fd)


class QDenseNet(nn.Module):
    def __init__(self, trained_block_list, fw_block_list, bw_block_list,
                    num_classes=10, depth=22, growthRate=12, compressionRate=2,
                    freeze_idxs=[]):
        super(QDenseNet, self).__init__()
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 6

        self.growthRate = growthRate
        self.in_planes = growthRate * 2

        layer_idxs = [i for i in range(1 + 1 + n*3 + 2)]
        self.max_idx = 1 + 1 + n*3 + 2 - 1
        assert set(freeze_idxs) <= set(layer_idxs), 'Invalid layer idxs'

        self._trained_idxs = [idx for idx in layer_idxs if idx not in freeze_idxs]
        assert set(self._trained_idxs) == set([i for i in range(max(self._trained_idxs) + 1 - len(self._trained_idxs), max(self._trained_idxs) + 1)]), \
             "No continous block of trained layers"

        self._trained_block = trained_block_list[0]
        self._trained_trans = trained_block_list[1]

        self._fw_block = fw_block_list[0]
        self._fw_trans = fw_block_list[1]

        self._bw_block = bw_block_list[0]
        self._bw_trans = bw_block_list[1]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, bias=False)

        if 0 not in self._trained_idxs:
            for parameter in self.conv1.parameters():
                parameter.requires_grad = False

        self._block_idx = 1

        self.dense1 = self._make_denseblock(n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(n)

        self.bn = nn.BatchNorm2d(self.in_planes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.in_planes, num_classes)

        # freeze linear layer in case it does not get trained
        if self.max_idx not in self._trained_idxs:
            for parameter in self.fc.parameters():
                parameter.requires_grad = False
            for parameter in self.bn.parameters():
                parameter.requires_grad = False

    def _make_denseblock(self, blocks):
        layers = []
        for _ in range(blocks):

            # case fw layer(frozen)
            if self._block_idx < min(self._trained_idxs):
                transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
                layers.append(self._fw_block(self.in_planes, growthRate=self.growthRate,
                                             is_transition=transition))

            # case bw+(fw) layer (also frozen)
            elif self._block_idx > max(self._trained_idxs):
                layers.append(self._bw_block(self.in_planes, growthRate=self.growthRate))

            # trained layer
            elif self._block_idx in self._trained_idxs:
                is_first = True if self._block_idx == min(self._trained_idxs) else False
                layers.append(self._trained_block(self.in_planes, growthRate=self.growthRate,
                                                  is_first=is_first))

            else: raise ValueError

            self.in_planes += self.growthRate
            self._block_idx += 1
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        in_planes = self.in_planes
        out_planes = int(math.floor(self.in_planes // compressionRate))
        self.in_planes = out_planes

        trans = None
        # case fw layer(frozen)
        if self._block_idx < min(self._trained_idxs):
            transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
            trans = self._fw_trans(in_planes, out_planes, is_transition=transition)

        # case bw+(fw) layer (also frozen)
        elif self._block_idx > max(self._trained_idxs):
            trans = self._bw_trans(in_planes, out_planes)

        # trained layer
        elif self._block_idx in self._trained_idxs:
            is_first = True if self._block_idx == min(self._trained_idxs) else False
            trans = self._trained_trans(in_planes, out_planes, is_first=is_first)
        else: raise ValueError

        self._block_idx += 1
        return trans

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class QDenseNet40(QDenseNet):
    def __init__(self, num_classes=10, freeze=[]):
        super(QDenseNet40, self).__init__([Bottleneck, Transition],
                                          [QFWBottleneck, QFWTransition],
                                          [QBWBottleneck, QBWTransition],
                                          num_classes=num_classes, depth=40, growthRate=12,
                                          compressionRate=2, freeze_idxs=freeze)
    @staticmethod
    def n_freezable_layers():
        return 1 + 1 + 3*6 + 2

    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qdensenet40, QDenseNet40.n_freezable_layers())
        return random.choice(configs)