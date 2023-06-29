'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.

from https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
'''
import torch.nn as nn
import torch.nn.functional as F


class IOLayer(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding):
        super(IOLayer, self).__init__()
        self.conv1 = nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.gn1 = nn.GroupNorm(output, output)
        self.relu1 = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        return x


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(planes, planes)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.gn2 = nn.GroupNorm(planes, planes)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(out_planes, out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(out_planes, out_planes)
            )

    def forward(self, x):
        out = self.relu1(self.gn1(self.conv1(x)))
        out = self.relu2(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2GroupNorm(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2GroupNorm, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.layer_in = IOLayer(3, 32, 3, 1, 1)
        self.layers = self._make_layers(in_planes=32)
        self.layer_out = IOLayer(320, 1280, 1, 1, 0)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer_in(x)
        out = self.layers(out)
        out = self.layer_out(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out