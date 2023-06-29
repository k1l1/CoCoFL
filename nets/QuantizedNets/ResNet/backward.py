import torch

from nets.QuantizedNets.utils.utils import filter_state_dict_keys
from nets.QuantizedNets.utils.backwards import QBWConv2dBN, BWConv2dBN


class QBWBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(QBWBasicBlock, self).__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.convbn1 = QBWConv2dBN(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.convbn2 = QBWConv2dBN(planes, planes, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_convbn = QBWConv2dBN(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
        else:
            self.shortcut_convbn = None

        ''' since QBWBasicBlock is only used for the part where no training is done,
        all params do not require a grad '''
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
                                filter_state_dict_keys(state_dict, prefix + 'bn2.running_var'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale_bw'))

        if len([key for key in state_dict.keys() if key.endswith(prefix + 'shortcut.0.weight')]) == 1:
            self.shortcut_convbn.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.0.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.bias'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_mean'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_var'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale_bw'))
        pass

    def forward(self, x):
        out = torch.nn.functional.relu(self.convbn1(x))
        out = self.convbn2(out)
        if self.shortcut_convbn is not None:
            x = self.shortcut_convbn(x)
        out += x
        out = torch.nn.functional.relu(out)
        return out


class QBWBottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(QBWBottleneck, self).__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.convbn1 = QBWConv2dBN(in_planes, planes, kernel_size=1, stride=1,
                                   padding=0)
        self.convbn2 = QBWConv2dBN(planes, planes, kernel_size=3, stride=stride,
                                   padding=1)
        self.convbn3 = QBWConv2dBN(planes, self.expansion*planes, kernel_size=1,
                                   stride=1, padding=0)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_convbn = QBWConv2dBN(in_planes, self.expansion*planes, kernel_size=1,
                                               stride=stride)
        else:
            self.shortcut_convbn = None

        '''Since QBWBottleneck is only used for the part where no backward w.r.t \
            parameters is required, all parameters do not require a grad.'''
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
                                filter_state_dict_keys(state_dict, prefix + 'bn2.running_var'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale_bw'))

        self.convbn3.init(filter_state_dict_keys(state_dict, prefix + 'conv3.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'bn3.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'bn3.bias'),
                                filter_state_dict_keys(state_dict, prefix + 'bn3.running_mean'),
                                filter_state_dict_keys(state_dict, prefix + 'bn3.running_var'),
                                filter_state_dict_keys(state_dict, prefix + 'bn3.op_scale'),
                                filter_state_dict_keys(state_dict, prefix + 'bn3.op_scale_bw'))

        if len([key for key in state_dict.keys() if key.endswith(prefix + 'shortcut.0.weight')]) == 1:
            self.shortcut_convbn.init(filter_state_dict_keys(state_dict, prefix + 'shortcut.0.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.weight'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.bias'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_mean'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.running_var'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale'),
                                      filter_state_dict_keys(state_dict, prefix + 'shortcut.1.op_scale_bw'))
        pass

    def forward(self, x):
        out = torch.nn.functional.relu(self.convbn1(x))
        out = torch.nn.functional.relu(self.convbn2(out))
        out = self.convbn3(out)
        if self.shortcut_convbn is not None:
            x = self.shortcut_convbn(x)
        out += x
        out = torch.nn.functional.relu(out)
        return out