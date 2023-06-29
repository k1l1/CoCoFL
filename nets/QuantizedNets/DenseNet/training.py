import torch.nn as nn
import torch.nn.functional as F

from nets.QuantizedNets.utils.training import Cat, BatchNorm2d, Conv2d


class Bottleneck(nn.Module):
    persistant_buffers = True
    track_running_stats = True

    def __init__(self, in_planes, expansion=4, growthRate=12, is_transition=False,
                        is_first=False):
        super(Bottleneck, self).__init__()

        planes = expansion * growthRate

        if is_first:
            self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        else:
            self.bn1 = BatchNorm2d(in_planes, track_running_stats=self.track_running_stats,
                                   persistant_buffers=self.persistant_buffers)

        self.relu1 = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2d(planes, track_running_stats=self.track_running_stats,
                                persistant_buffers=self.persistant_buffers)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = Conv2d(planes, growthRate, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.cat = Cat(persistant_buffers=self.persistant_buffers)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv2(out)
        out = self.cat([x, out], dim=1)

        return out


class Transition(nn.Module):
    persistant_buffers = True
    track_running_stats = True

    def __init__(self, in_planes, out_planes, is_transition=False, is_first=False):
        super(Transition, self).__init__()

        if is_first:
            self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        else:
            self.bn1 = BatchNorm2d(in_planes, track_running_stats=self.track_running_stats,
                                   persistant_buffers=self.persistant_buffers)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out