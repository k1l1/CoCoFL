import torch.nn as nn

from nets.QuantizedNets.MobileNet.training import Block
from nets.QuantizedNets.utils.backwards import QBWConv2dBN, BWConv2dBN, QBWConv2d, BWConv2d
from nets.QuantizedNets.utils.utils import filter_state_dict_keys


class QBWBlock(nn.Module):

    def __init__(self, in_planes, out_planes, expansion, stride, is_transition=False):
        super(QBWBlock, self).__init__()
        self.stride = stride

        self._register_load_state_dict_pre_hook(self.sd_hook)

        planes = expansion * in_planes

        self.convbn1 = QBWConv2dBN(in_planes, planes, kernel_size=1, stride=1, padding=0)
        self.convbn2 = BWConv2dBN(planes, planes, kernel_size=3, stride=stride,
                                            padding=1, groups=planes)
        self.convbn3 = QBWConv2dBN(planes, out_planes, kernel_size=1, stride=1, padding=0)

        if stride == 1 and in_planes != out_planes:
            self.shortcut_convbn = QBWConv2dBN(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut_convbn = None

        '''Since QBWBlock is only used in the forward pass no backprop w.r.t. parameters is required'''
        for parameter in self.parameters():
            parameter.requires_grad = False

    def sd_hook(self, state_dict, prefix, *_):
        self.convbn1.init(filter_state_dict_keys(state_dict, prefix + 'conv1.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'bn1.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'bn1.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'bn1.running_mean'),
                            filter_state_dict_keys(state_dict, prefix + 'bn1.running_var'),
                            filter_state_dict_keys(state_dict, prefix + 'bn1.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'bn1.op_scale_bw'))

        self.convbn2.init(filter_state_dict_keys(state_dict, prefix + 'conv2.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'bn2.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'bn2.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'bn2.running_mean'),
                            filter_state_dict_keys(state_dict, prefix + 'bn2.running_var'))

        self.convbn3.init(filter_state_dict_keys(state_dict, prefix + 'conv3.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'bn3.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'bn3.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'bn3.running_mean'),
                            filter_state_dict_keys(state_dict, prefix + 'bn3.running_var'),
                            filter_state_dict_keys(state_dict, prefix + 'bn3.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'bn3.op_scale_bw'))

        if self.shortcut_convbn is not None:
            self.shortcut_convbn.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.0.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.bias'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_mean'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_var'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale_bw'))

    def forward(self, x):
        out = nn.functional.relu(self.convbn1(x))
        out = nn.functional.relu(self.convbn2(out))
        out = self.convbn3(out)

        if self.shortcut_convbn is not None:
            x = self.shortcut_convbn(x)

        if self.stride == 1:
            out += x
        return out


class QBWBlockGroupNorm(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride, is_transition=False):
        super(QBWBlockGroupNorm, self).__init__()
        self.stride = stride

        self._register_load_state_dict_pre_hook(self.sd_hook)

        planes = expansion * in_planes

        self.conv1 = QBWConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(planes, planes)

        self.conv2 = BWConv2d(planes, planes, kernel_size=3, stride=stride,
                                            padding=1, groups=planes)

        self.gn2 = nn.GroupNorm(planes, planes)
        self.conv3 = QBWConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.gn3 = nn.GroupNorm(out_planes, out_planes)

        if stride == 1 and in_planes != out_planes:
            self.shortcut_conv = QBWConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
            self.shortcut_gn = nn.GroupNorm(out_planes, out_planes)
        else:
            self.shortcut_conv = None

        '''Since QBWBlock is only used in the forward pass no backprop w.r.t. parameters is required'''
        for parameter in self.parameters():
            parameter.requires_grad = False

    def sd_hook(self, state_dict, prefix, *_):
        self.conv1.init(filter_state_dict_keys(state_dict, prefix + 'conv1.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'conv1.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'conv1.op_scale_bw'))

        self.conv2.init(filter_state_dict_keys(state_dict, prefix + 'conv2.weight'))

        self.conv3.init(filter_state_dict_keys(state_dict, prefix + 'conv3.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'conv3.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'conv3.op_scale_bw'))

        if self.shortcut_conv is not None:
            self.shortcut_conv.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.0.weight'),
                                    filter_state_dict_keys(state_dict, prefix + 'shortcut.0.op_scale'),
                                    filter_state_dict_keys(state_dict, prefix + 'shortcut.0.op_scale_bw'))
        pass

    def forward(self, x):
        out = nn.functional.relu(self.gn1(self.conv1(x)))
        out = nn.functional.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))

        if self.shortcut_conv is not None:
            x = self.shortcut_conv(x)
            x = self.shortcut_gn(x)

        if self.stride == 1:
            out += x
        return out