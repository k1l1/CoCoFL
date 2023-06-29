import torch.nn as nn
from nets.QuantizedNets.utils.training import Add, BatchNorm2d, GroupNorm, Conv2d


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    persistant_buffers = True
    track_running_stats = True

    def __init__(self, in_planes, out_planes, expansion, stride, is_transition=False):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(planes, track_running_stats=self.track_running_stats, momentum=1.0,
                                    persistant_buffers=self.persistant_buffers)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = BatchNorm2d(planes, track_running_stats=self.track_running_stats, momentum=1.0,
                                persistant_buffers=self.persistant_buffers)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BatchNorm2d(out_planes, track_running_stats=self.track_running_stats, momentum=1.0,
                                    persistant_buffers=self.persistant_buffers)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                BatchNorm2d(out_planes, track_running_stats=self.track_running_stats, momentum=1.0,
                                persistant_buffers=self.persistant_buffers)
            )

        if stride == 1:
            self.add = Add(persistant_buffers=self.persistant_buffers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride == 1:
            return self.add(out, self.shortcut(x))
        else:
            return out


class BlockGroupNorm(nn.Module):
    '''expand + depthwise + pointwise'''

    persistant_buffers = True
    track_running_stats = True

    def __init__(self, in_planes, out_planes, expansion, stride, is_transition=False):
        super(BlockGroupNorm, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.gn1 = GroupNorm(planes, planes, persistant_buffers=self.persistant_buffers)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, dilation=1, bias=False)
        self.gn2 = GroupNorm(planes, planes, persistant_buffers=self.persistant_buffers)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.gn3 = GroupNorm(out_planes, out_planes, persistant_buffers=self.persistant_buffers)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                GroupNorm(out_planes, out_planes, persistant_buffers=self.persistant_buffers))

        if stride == 1:
            self.add = Add(persistant_buffers=self.persistant_buffers)

    def forward(self, x):
        out = self.relu1(self.gn1(self.conv1(x)))
        out = self.relu2(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.stride == 1:
            return self.add(out, self.shortcut(x))
        else:
            return out