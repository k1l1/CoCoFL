import torch

from nets.QuantizedNets.utils.utils import filter_state_dict_keys, tensor_scale
from nets.QuantizedNets.utils.forward import QConv2dBN, QConv2dBNRelu, QAddRelu


class QFWBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_transition=False):
        super(QFWBasicBlock, self).__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.convbn1relu = QConv2dBNRelu(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.convbn2 = QConv2dBN(planes, planes, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_convbn = QConv2dBN(in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut_convbn = None

        self.add_relu = QAddRelu()

        '''Since QBasicBlock is only used for the part where no backward
        w.r.t to params is required, all params do not require a grad'''
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

        self.convbn2.init(filter_state_dict_keys(state_dict, prefix + 'conv2.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.bias'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.running_mean'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.running_var'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale'))

        if len([key for key in state_dict.keys() if key.endswith(prefix + 'shortcut.0.weight')]) == 1:
            self.shortcut_convbn.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.0.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.bias'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_mean'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_var'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale'))
        self.add_relu.init(filter_state_dict_keys(state_dict, prefix + 'add.op_scale'))
        pass

    def forward(self, x):
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)

        out = self.convbn1relu(x)
        out = self.convbn2(out)

        if self.shortcut_convbn is not None:
            x = self.shortcut_convbn(x)
        out = self.add_relu(out, x)

        if self._is_transition:
            out = torch.dequantize(out)
        return out


class QFWBottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_transition=False):
        super(QFWBottleneck, self).__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.convbn1relu = QConv2dBNRelu(in_planes, planes, kernel_size=1, stride=1,
                                            padding=0)
        self.convbn2relu = QConv2dBNRelu(planes, planes, kernel_size=3, stride=stride,
                                            padding=1)
        self.convbn3 = QConv2dBN(planes, self.expansion*planes, kernel_size=1, stride=1,
                                    padding=0)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_convbn = QConv2dBN(in_planes, self.expansion*planes, kernel_size=1,
                                             stride=stride)
        else:
            self.shortcut_convbn = None

        self.add_relu = QAddRelu()

        '''Since QFBottleneck is only used for the part where no backward w.r.t params \
            is required, all parameters do not require a grad'''
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

        if len([key for key in state_dict.keys() if key.endswith(prefix + 'shortcut.0.weight')]) == 1:
            self.shortcut_convbn.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.0.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.bias'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_mean'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_var'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale'))
        self.add_relu.init(filter_state_dict_keys(state_dict, prefix + 'add.op_scale'))
        pass

    def forward(self, x):
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)

        out = self.convbn1relu(x)
        out = self.convbn2relu(out)
        out = self.convbn3(out)

        if self.shortcut_convbn is not None:
            x = self.shortcut_convbn(x)
        out = self.add_relu(out, x)

        if self._is_transition:
            out = torch.dequantize(out)
        return out