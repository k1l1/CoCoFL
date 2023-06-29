import torch.nn as nn

from nets.QuantizedNets.utils.training import Add, BatchNorm2d


class InputLayer(nn.Module):
    def __init__(self):
        super(InputLayer, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    persistant_buffers = True
    track_running_stats = True

    def __init__(self, in_planes, planes, stride=1, is_transition=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = BatchNorm2d(planes, track_running_stats=self.track_running_stats, momentum=1.0,
                                persistant_buffers=self.persistant_buffers)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn2 = BatchNorm2d(planes, track_running_stats=self.track_running_stats, momentum=1.0,
                                persistant_buffers=self.persistant_buffers)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes, track_running_stats=self.track_running_stats, momentum=1.0,
                                    persistant_buffers=self.persistant_buffers)
            )
        self.add = Add(persistant_buffers=self.persistant_buffers)
        self.relu2 = nn.ReLU(inplace=False)
        return

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        x = self.shortcut(x)

        out = self.add(x, out)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    persistant_buffers = True
    track_running_stats = True

    def __init__(self, in_planes, planes, stride=1, is_transition=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, track_running_stats=self.track_running_stats,
                               persistant_buffers=self.persistant_buffers)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, track_running_stats=self.track_running_stats,
                                persistant_buffers=self.persistant_buffers)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(self.expansion*planes, track_running_stats=self.track_running_stats,
                               persistant_buffers=self.persistant_buffers)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes, track_running_stats=self.track_running_stats,
                            persistant_buffers=self.persistant_buffers)
            )
        self.add = Add(persistant_buffers=self.persistant_buffers)
        self.relu3 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)

        out = self.add(x, out)
        out = self.relu3(out)
        return out
