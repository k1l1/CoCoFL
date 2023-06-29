import torch
import torch.nn as nn
from nets.QuantizedNets.utils.utils import filter_state_dict_keys
from nets.QuantizedNets.utils.backwards import QBWConv2dBN, QBWConv2d


class QBWBottleneck(torch.nn.Module):
    def __init__(self, in_planes, expansion=4, growthRate=12, is_transition=False,
                        is_first=False):
        super(QBWBottleneck, self).__init__()

        planes = expansion * growthRate
        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.convbn1 = QBWConv2dBN(in_planes, planes, kernel_size=1, stride=1, padding=0)
        self.conv2 = QBWConv2d(planes, growthRate, kernel_size=3, stride=1, padding=1)

        ''' since QBWBottleneck is only used for the part where no training is done,
        all params do not require a grad '''
        for parameter in self.parameters():
            parameter.requires_grad = False

    def sd_hook(self, state_dict, prefix, *_):
        self.convbn1.init(filter_state_dict_keys(state_dict, prefix + 'conv1.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.bias'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.running_mean'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.running_var'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale_bw'))

        self.conv2.init(filter_state_dict_keys(state_dict, prefix + 'conv2.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'conv2.op_scale'),
                                filter_state_dict_keys(state_dict, prefix + 'conv2.op_scale_bw'))

    def forward(self, x):
        out = self.bn1(x)
        out = nn.functional.relu(out)
        out = self.convbn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = torch.cat([x, out], dim=1)
        return out


class QBWTransition(nn.Module):
    def __init__(self, in_planes, out_planes, is_transition=False, is_first=False):
        super(QBWTransition, self).__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.conv1 = QBWConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)

        ''' since QBWTransition is only used for the part where no training is done,
        all params do not require a grad '''
        for parameter in self.parameters():
            parameter.requires_grad = False

    def sd_hook(self, state_dict, prefix, *_):
        self.conv1.init(filter_state_dict_keys(state_dict, prefix + 'conv1.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'conv1.op_scale'),
                                filter_state_dict_keys(state_dict, prefix + 'conv1.op_scale_bw'))

    def forward(self, x):
        out = self.bn1(x)
        out = nn.functional.relu(out)
        out = self.conv1(out)
        out = nn.functional.avg_pool2d(out, 2)
        return out
