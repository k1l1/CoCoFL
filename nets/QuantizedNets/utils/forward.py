import torch

from nets.QuantizedNets.utils.utils import prepack_conv2d, fuse_conv_bn_weights, tensor_scale
from nets.QuantizedNets.utils.utils import prepack_linear


class QConv2dBN():
    def __init__(self, in_channels, out_channels, kernel_size=3,
                    stride=1, padding=0, groups=1):

        self._weight_shape = [out_channels, in_channels//groups, kernel_size, kernel_size]

        self._stride = stride
        self._padding = padding
        self._groups = groups

        self._is_initialized = False

    def init(self, weight, bn_weight, bn_bias, run_mean, run_var, op_scale):

        assert list(weight.shape) == self._weight_shape, f'Expected weight shape is wrong \
                expected {self._weight_shape}, got {list(weight.shape)}'

        f_w, f_bias = fuse_conv_bn_weights(weight, None, run_mean, run_var,
                                            1e-5, bn_weight, bn_bias)

        q_f_w = torch.quantize_per_tensor(f_w, tensor_scale(f_w), 0, dtype=torch.qint8)
        self._prepack = prepack_conv2d(q_f_w, f_bias, self._stride, self._padding,
                                        groups=self._groups)
        self._op_scale = op_scale
        self._is_initialized = True

    def __call__(self, x):
        assert self._is_initialized, f"Error: {self} not initialized"
        return torch.ops.quantized.conv2d(x, self._prepack, self._op_scale, 64)


class QConv2d(QConv2dBN):

    #!overrides
    def init(self, weight, op_scale):
        assert list(weight.shape) == self._weight_shape, f'Expected weight shape is wrong \
                expected {self._weight_shape}, got {list(weight.shape)}'

        q_w = torch.quantize_per_tensor(weight, tensor_scale(weight), 0, dtype=torch.qint8)
        self._prepack = prepack_conv2d(q_w, None, self._stride, self._padding, groups=self._groups)

        self._op_scale = op_scale
        self._is_initialized = True


class QConv2dBNRelu(QConv2dBN):

    #!overrides
    def __call__(self, x):
        assert self._is_initialized, f"Error: {self} not initialized"
        return torch.ops.quantized.conv2d_relu(x, self._prepack, self._op_scale, 64)


class QBatchNorm2d():
    def __init__(self, channels):
        self._is_initialized = False
        self._weight_shape = channels

    def init(self, weight, bias, running_mean, running_var, op_scale):
        assert self._weight_shape == weight.shape[0],  f'Expected weight shape is wrong \
                expected {self._weight_shape}, got {list(weight.shape)}'
        self._weight = weight
        self._bias = bias
        self._running_mean = running_mean
        self._running_var = running_var
        self._op_scale = op_scale

        self._is_initialized = True

    def __call__(self, x):
        assert self._is_initialized, f"Error: {self} not initialized"
        return torch.ops.quantized.batch_norm2d(x, self._weight, self._bias, self._running_mean, self._running_var,
                                                1e-5, self._op_scale, 64)


class QBatchNorm2drelu(QBatchNorm2d):
    def __call__(self, x):
        assert self._is_initialized, f"Error: {self} not initialized"
        return torch.ops.quantized.batch_norm2d_relu(x, self._weight, self._bias, self._running_mean, self._running_var,
                                                     1e-5, self._op_scale, 64)


class QAdd():
    def __init__(self):
        self._is_initialized = False

    def init(self, op_scale):
        self._op_scale = op_scale
        self._is_initialized = True

    def __call__(self, x1, x2):
        assert self._is_initialized, f"Error: {self} not initialized"
        return torch.ops.quantized.add(x1, x2, self._op_scale, 64)


class QAddRelu(QAdd):
    def __call__(self, x1, x2):
        assert self._is_initialized, f"Error: {self} not initialized"
        return torch.ops.quantized.add_relu(x1, x2, self._op_scale, 64)


class QCat():
    def __init__(self):
        self._is_initialized = False

    def init(self, op_scale):
        self._op_scale = op_scale
        self._is_initialized = True

    def __call__(self, list_of_tensors, dim=0):
        assert self._is_initialized, f"Error: {self} not initiaized"
        return torch.ops.quantized.cat(list_of_tensors, dim, self._op_scale, 64)


class QLinear():
    def __init__(self, in_features, out_features):
        self._weight_shape = [out_features, in_features]

    def init(self, weight, bias, op_scale):
        assert list(weight.shape) == self._weight_shape, f'Expected weight shape is wrong \
            expected {self._weight_shape}, got {list(weight.shape)}'

        qw = torch.quantize_per_tensor(weight, tensor_scale(weight), 0, dtype=torch.qint8)
        self._prepack = prepack_linear(qw, bias)

        self._op_scale = op_scale

        self._is_initialized = True

    def __call__(self, x):
        assert self._is_initialized, f"Error: {self} not initialized"
        return torch.ops.quantized.linear(x, self._prepack, self._op_scale, 64)


class QLinearReLU(QLinear):
    def __call__(self, x):
        assert self._is_initialized, f"Error: {self} not initalized"
        return torch.ops.quantized.linear_relu(x, self._prepack, self._op_scale, 64)


class QLayerNorm():
    def __init__(self, normalized_shape):
        self._normalized_shape = normalized_shape

    def init(self, weight, bias, op_scale):
        self.w = weight
        self.b = bias
        self._op_scale = op_scale

        self._is_initialized = True

    def __call__(self, x):
        assert self._is_initialized, f"Error: {self} not initialized"
        return torch.ops.quantized.layer_norm(x, [self._normalized_shape], self.w, self.b, 1e-5, self._op_scale, 64)


class QGroupNorm():
    def __init__(self, num_groups, num_channels):
        self._num_groups = num_groups
        self._num_channels = num_channels
        self._is_initialized = False

    def init(self, weight, bias, op_scale):
        self.w = weight
        self.b = bias
        self._op_scale = op_scale
        self._is_initialized = True

    def __call__(self, x):
        assert self._is_initialized, f"Error: {self} not initalized"
        return torch.ops.quantized.group_norm(x, self._num_groups, self.w, self.b, 1e-5, self._op_scale, 64)
