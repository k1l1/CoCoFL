import torch
from torch.autograd import Function
from nets.QuantizedNets.utils.utils import prepack_conv2d, fuse_conv_bn_weights, \
                                            tensor_scale, prepack_conv2d_transpose, \
                                                prepack_linear


class QBWConv2dBN():
    def __init__(self, in_channels, out_channels, kernel_size=3,
                    stride=1, padding=0, groups=1):
        self._weight_shape = [out_channels, in_channels//groups, kernel_size, kernel_size]

        self._stride = stride
        self._padding = padding
        self._groups = groups

        self._is_initialized = False
        self._qdict = {}

    def init(self, weight, bn_weight, bn_bias, run_mean, run_var, op_scale, op_scale_bw):
        assert list(weight.shape) == self._weight_shape, f"Expected weight shape is wrong \
                expected {self._weight_shape}, got {list(weight.shape)}"

        # fusion of weights and BN
        f_w, f_bias = fuse_conv_bn_weights(weight, None, run_mean, run_var,
                                            1e-5, bn_weight, bn_bias)

        # prepack for forward pass
        q_f_w = torch.quantize_per_tensor(f_w, tensor_scale(f_w), 0, dtype=torch.qint8)
        self._qdict["prepack"] = prepack_conv2d(q_f_w, f_bias, self._stride, self._padding,
                                                groups=self._groups)
        self._qdict["op_scale"] = op_scale

        # prepack for backward pass
        self._qdict["prepack_transposed"] = prepack_conv2d_transpose(q_f_w, None, self._stride, self._padding, groups=self._groups)
        self._qdict["op_scale_bw"] = op_scale_bw

        self._is_initialized = True

    def __call__(self, x):
        return qconv2d_function.apply(x, self._qdict)


class QBWConv2d(QBWConv2dBN):
    def init(self, weight, op_scale, op_scale_bw):
        assert list(weight.shape) == self._weight_shape, f"Expected weight shape is wrong \
                expected {self._weight_shape}, got {list(weight.shape)}"

        # prepack for forward pass
        q_w = torch.quantize_per_tensor(weight, tensor_scale(weight), 0, dtype=torch.qint8)
        self._qdict["prepack"] = prepack_conv2d(q_w, None, self._stride, self._padding,
                                                groups=self._groups)
        self._qdict["op_scale"] = op_scale

        # prepack for backward pass
        self._qdict["prepack_transposed"] = prepack_conv2d_transpose(q_w, None, self._stride, self._padding, groups=self._groups)
        self._qdict["op_scale_bw"] = op_scale_bw

        self._is_initialized = True


class BWConv2dBN():
    def __init__(self, in_channels, out_channels, kernel_size=3,
                    stride=1, padding=0, groups=1):
        self._weight_shape = [out_channels, in_channels//groups, kernel_size, kernel_size]

        self._stride = stride
        self._padding = padding
        self._groups = groups

        self._is_initialized = False

    def init(self, weight, bn_weight, bn_bias, run_mean, run_var):
        assert list(weight.shape) == self._weight_shape, f"Expected weight shape is wrong \
            expected {self._weight_shape}, got {list(weight.shape)}"

        # fusion of weights and BN
        self.f_w, self.f_bias = fuse_conv_bn_weights(weight, None, run_mean, run_var,
                                                     1e-5, bn_weight, bn_bias)
        self._is_initialized = True

    def __call__(self, x):
        return torch.nn.functional.conv2d(x, self.f_w, bias=self.f_bias, stride=self._stride,
                                            padding=self._padding, dilation=1, groups=self._groups)


class BWConv2d():
    def __init__(self, in_channels, out_channels, kernel_size=3,
                    stride=1, padding=0, groups=1):

        self._weight_shape = [out_channels, in_channels//groups, kernel_size, kernel_size]
        self._stride = stride
        self._padding = padding
        self._groups = groups
        self._is_initialized = False

    def init(self, weight):
        assert list(weight.shape) == self._weight_shape, f"Expected weight shape is wrong \
            expected {self._weight_shape}, got {list(weight.shape)}"

        self.f_w = weight
        self._is_initialized = True

    def __call__(self, x):
        return torch.nn.functional.conv2d(x, self.f_w, stride=self._stride,
                                            padding=self._padding, dilation=1, groups=self._groups)


class QBWLinear():
    def __init__(self, in_features, out_features):
        self._weight_shape = [out_features, in_features]
        self._qdict = {}
        self._is_initialized = False

    def init(self, weight, bias, op_scale, op_scale_bw):
        assert list(weight.shape) == self._weight_shape, f"Expected weight shape is wrong \
            expected {self._weight_shape}, got {list(weight.shape)}"

        # prepack  for forward pass
        q_w = torch.quantize_per_tensor(weight, tensor_scale(weight), 0, dtype=torch.qint8)
        self._qdict["prepack"] = prepack_linear(q_w, bias)
        self._qdict["weight"] = weight
        self._qdict["prepack_bw"] = prepack_linear(q_w.t())
        self._qdict["op_scale"] = op_scale
        self._qdict["op_scale_bw"] = op_scale_bw

        self._is_initialized = True

    def __call__(self, x):
        return qlinear_function.apply(x, self._qdict)


class qconv2d_function(Function):
    @staticmethod
    def forward(ctx, input, qdict):
        ctx.set_materialize_grads(False)

        ctx.op_scale_bw = qdict["op_scale_bw"]
        ctx.prepack_transposed = qdict["prepack_transposed"]

        input = torch.quantize_per_tensor(input, tensor_scale(input), 64, torch.quint8)
        out = torch.ops.quantized.conv2d(input, qdict["prepack"], qdict["op_scale"], 64)
        return torch.dequantize(out)

    @staticmethod
    def backward(ctx, grad_ouput):
        if grad_ouput is None: return None, None

        grad_ouput = torch.quantize_per_tensor(grad_ouput, tensor_scale(grad_ouput), 64, torch.quint8)
        grad_input = torch.ops.quantized.conv_transpose2d(grad_ouput, ctx.prepack_transposed, ctx.op_scale_bw, 64)
        return torch.dequantize(grad_input), None


class qlinear_function(Function):
    @staticmethod
    def forward(ctx, input, qdict):
        ctx.set_materialize_grads(False)

        ctx.prepack_bw = qdict["prepack_bw"]
        ctx.op_scale_bw = qdict["op_scale_bw"]

        input = torch.quantize_per_tensor(input, tensor_scale(input), 64,  dtype=torch.quint8)
        out = torch.ops.quantized.linear(input, qdict["prepack"], qdict["op_scale"], 64)
        return torch.dequantize(out)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None: return None, None
        grad_output = torch.quantize_per_tensor(grad_output, tensor_scale(grad_output[0, 0, :]), 64, torch.quint8)
        grad_input = torch.ops.quantized.linear(grad_output, ctx.prepack_bw, ctx.op_scale_bw, 64)
        return torch.dequantize(grad_input), None