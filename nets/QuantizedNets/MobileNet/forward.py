import torch
import torch.nn as nn

from nets.QuantizedNets.utils.forward import QConv2dBN, QConv2dBNRelu, QAdd, QConv2d, QGroupNorm
from nets.QuantizedNets.utils.utils import filter_state_dict_keys, tensor_scale


class QFWBlock(torch.nn.Module):

    def __init__(self, in_planes, out_planes, expansion, stride, is_transition=False):
        super(QFWBlock, self).__init__()
        self.stride = stride

        self._register_load_state_dict_pre_hook(self.sd_hook)

        planes = expansion * in_planes

        self.convbn1relu = QConv2dBNRelu(in_planes, planes, kernel_size=1, stride=1, padding=0)

        self.convbn2relu = QConv2dBNRelu(planes, planes, kernel_size=3, stride=stride,
                                         padding=1, groups=planes)

        self.convbn3 = QConv2dBN(planes, out_planes, kernel_size=1, stride=1, padding=0)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut_convbn = QConv2dBN(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut_convbn = None
        if stride == 1:
            self.add = QAdd()
        else:
            self.add = None

        '''Since QFWBlock is only used in the forward pass no backprop w.r.t. parameters is required'''
        for parameter in self.parameters():
            parameter.requires_grad = False

        self._is_transition = is_transition

    def sd_hook(self, state_dict, prefix, *_):
        self.convbn1relu.init(filter_state_dict_keys(state_dict, prefix + 'conv1.weight'),
                              filter_state_dict_keys(state_dict, prefix + 'bn1.weight'),
                              filter_state_dict_keys(state_dict, prefix + 'bn1.bias'),
                              filter_state_dict_keys(state_dict, prefix + 'bn1.running_mean'),
                              filter_state_dict_keys(state_dict, prefix + 'bn1.running_var'),
                              filter_state_dict_keys(state_dict, prefix + 'bn1.op_scale'))

        self.convbn2relu.init(filter_state_dict_keys(state_dict, prefix + 'conv2.weight'),
                              filter_state_dict_keys(state_dict, prefix + 'bn2.weight'),
                              filter_state_dict_keys(state_dict, prefix + 'bn2.bias'),
                              filter_state_dict_keys(state_dict, prefix + 'bn2.running_mean'),
                              filter_state_dict_keys(state_dict, prefix + 'bn2.running_var'),
                              filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale'))

        self.convbn3.init(filter_state_dict_keys(state_dict, prefix + 'conv3.weight'),
                             filter_state_dict_keys(state_dict, prefix + 'bn3.weight'),
                             filter_state_dict_keys(state_dict, prefix + 'bn3.bias'),
                             filter_state_dict_keys(state_dict, prefix + 'bn3.running_mean'),
                             filter_state_dict_keys(state_dict, prefix + 'bn3.running_var'),
                             filter_state_dict_keys(state_dict, prefix + 'bn3.op_scale'))

        if self.shortcut_convbn is not None:
            self.shortcut_convbn.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.0.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.bias'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_mean'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_var'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale'))
        if self.add is not None:
            self.add.init(filter_state_dict_keys(state_dict, prefix + 'add.op_scale'))

    def forward(self, x):
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)
        out = self.convbn1relu(x)
        out = self.convbn2relu(out)
        out = self.convbn3(out)
        if self.stride == 1:
            out = self.add(out, x if self.shortcut_convbn is None else self.shortcut_convbn(x))

        if self._is_transition:
            out = torch.dequantize(out)
        return out


class QFWBlockGroupNorm(torch.nn.Module):

    def __init__(self, in_planes, out_planes, expansion, stride, is_transition=False):
        super(QFWBlockGroupNorm, self).__init__()
        self.stride = stride

        self._register_load_state_dict_pre_hook(self.sd_hook)

        planes = expansion * in_planes

        self.conv1 = QConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)
        self.gn1 = QGroupNorm(planes, planes)

        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride,
                                        padding=1, groups=planes)
        self.gn2 = QGroupNorm(planes, planes)

        self.conv3 = QConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.gn3 = QGroupNorm(out_planes, out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut_conv = QConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
            self.shortcut_gn = QGroupNorm(out_planes, out_planes)
        else:
            self.shortcut_conv = None
        if stride == 1:
            self.add = QAdd()
        else:
            self.add = None

        '''Since QFWBlock is only used in the forward pass no backprop w.r.t. parameters is required'''
        for parameter in self.parameters():
            parameter.requires_grad = False

        self._is_transition = is_transition

    def sd_hook(self, state_dict, prefix, *_):
        self.conv1.init(filter_state_dict_keys(state_dict, prefix + 'conv1.weight'),
                             filter_state_dict_keys(state_dict, prefix + 'conv1.op_scale'))
        self.gn1.init(filter_state_dict_keys(state_dict, prefix + 'gn1.weight'),
                      filter_state_dict_keys(state_dict, prefix + 'gn1.bias'),
                      filter_state_dict_keys(state_dict, prefix + 'gn1.op_scale'))

        self.conv2.init(filter_state_dict_keys(state_dict, prefix + 'conv2.weight'),
                             filter_state_dict_keys(state_dict, prefix + 'conv2.op_scale'))
        self.gn2.init(filter_state_dict_keys(state_dict, prefix + 'gn2.weight'),
                      filter_state_dict_keys(state_dict, prefix + 'gn2.bias'),
                      filter_state_dict_keys(state_dict, prefix + 'gn2.op_scale'))

        self.conv3.init(filter_state_dict_keys(state_dict, prefix + 'conv3.weight'),
                             filter_state_dict_keys(state_dict, prefix + 'conv3.op_scale'))

        self.gn3.init(filter_state_dict_keys(state_dict, prefix + 'gn3.weight'),
                      filter_state_dict_keys(state_dict, prefix + 'gn3.bias'),
                      filter_state_dict_keys(state_dict, prefix + 'gn3.op_scale'))

        if self.shortcut_conv is not None:
            self.shortcut_conv.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.0.weight'),
                                    filter_state_dict_keys(state_dict, prefix + 'shortcut.0.op_scale'))
            self.shortcut_gn.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.1.weight'),
                                  filter_state_dict_keys(state_dict, prefix + 'shortcut.1.bias'),
                                  filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale'))
        if self.add is not None:
            self.add.init(filter_state_dict_keys(state_dict, prefix + 'add.op_scale'))
        pass

    def forward(self, x):
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)
        out = torch.nn.functional.relu(self.gn1(self.conv1(x)))
        out = torch.nn.functional.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.stride == 1:
            out = self.add(out, x if self.shortcut_conv is None else self.shortcut_gn(self.shortcut_conv(x)))

        if self._is_transition:
            out = torch.dequantize(out)
        return out