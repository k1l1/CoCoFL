import torch
from nets.QuantizedNets.utils.forward import QConv2dBNRelu, QConv2d, QCat, QBatchNorm2drelu
from nets.QuantizedNets.utils.utils import tensor_scale, filter_state_dict_keys


class QFWBottleneck(torch.nn.Module):
    def __init__(self, in_planes, expansion=4, growthRate=12, is_transition=False,
                        is_first=False):
        super(QFWBottleneck, self).__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)
        self._is_transition = is_transition
        planes = expansion * growthRate

        self.bn1relu = QBatchNorm2drelu(in_planes)

        self.convbn1relu = QConv2dBNRelu(in_planes, planes, kernel_size=1, stride=1,
                                         padding=0)
        self.conv2 = QConv2d(planes, growthRate, kernel_size=3, padding=1, stride=1)

        self.cat = QCat()

        '''Since QFWBottleneck is only used for the part where no backward
        w.r.t to params is required, all params do not require a grad'''
        for parameter in self.parameters():
            parameter.requires_grad = False

        self._is_transition = is_transition

    def sd_hook(self, state_dict, prefix, *_):
        self.bn1relu.init(filter_state_dict_keys(state_dict, prefix + 'bn1.weight'),
                          filter_state_dict_keys(state_dict, prefix + 'bn1.bias'),
                          filter_state_dict_keys(state_dict, prefix + 'bn1.running_mean'),
                          filter_state_dict_keys(state_dict, prefix + 'bn1.running_var'),
                          filter_state_dict_keys(state_dict, prefix + 'bn1.op_scale'))

        self.convbn1relu.init(filter_state_dict_keys(state_dict, prefix + 'conv1.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.bias'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.running_mean'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.running_var'),
                                filter_state_dict_keys(state_dict, prefix + 'bn2.op_scale'))

        self.conv2.init(filter_state_dict_keys(state_dict, prefix + 'conv2.weight'),
                                filter_state_dict_keys(state_dict, prefix + 'conv2.op_scale'))

        self.cat.init(filter_state_dict_keys(state_dict, prefix + 'cat.op_scale'))

    def forward(self, x):
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)
        out = self.bn1relu(x)
        out = self.convbn1relu(out)
        out = self.conv2(out)
        out = self.cat([x, out], 1)

        if self._is_transition:
            out = torch.dequantize(out)
        return out


class QFWTransition(torch.nn.Module):
    def __init__(self, in_planes, out_planes, is_transition=False):
        super(QFWTransition, self).__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.bn1relu = QBatchNorm2drelu(in_planes)
        self.conv1 = QConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)

        self._is_transition = is_transition

    def sd_hook(self, state_dict, prefix, *_):
        self.bn1relu.init(filter_state_dict_keys(state_dict, prefix + 'bn1.weight'),
                          filter_state_dict_keys(state_dict, prefix + 'bn1.bias'),
                          filter_state_dict_keys(state_dict, prefix + 'bn1.running_mean'),
                          filter_state_dict_keys(state_dict, prefix + 'bn1.running_var'),
                          filter_state_dict_keys(state_dict, prefix + 'bn1.op_scale'))

        self.conv1.init(filter_state_dict_keys(state_dict, prefix + 'conv1.weight'),
                        filter_state_dict_keys(state_dict, prefix + 'conv1.op_scale'))
        pass

    def forward(self, x):
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)
        out = self.bn1relu(x)
        out = self.conv1(out)
        out = torch.nn.functional.avg_pool2d(out, 2)

        if self._is_transition:
            out = torch.dequantize(out)
        return out
