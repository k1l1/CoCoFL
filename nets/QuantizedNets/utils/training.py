import torch
from torch.nn import BatchNorm2d as BN
from torch.nn import LayerNorm as LN
from torch.nn import GroupNorm as GN
from torch.nn import Conv2d as CV2d
from torch.nn import Linear as Lnr
from nets.QuantizedNets.utils.utils import tensor_scale
from torch.autograd import Function


class BatchNorm2d(BN):
    def __init__(self, num_features, eps=0.00001,
                 momentum=0.1, affine=True, track_running_stats=True, persistant_buffers=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats=track_running_stats)

        self.register_buffer('op_scale', 0.1*torch.ones(1), persistent=persistant_buffers)
        self.register_buffer('op_scale_bw', 0.1*torch.ones(1), persistent=persistant_buffers)

    def forward(self, x):
        x = gradient_observer.apply(x, self.op_scale_bw)
        x = super().forward(x)
        self.op_scale = torch.atleast_1d(torch.tensor(tensor_scale(x.detach())))
        return x


class LayerNorm(LN):
    def __init__(self, normalized_shape, persistant_buffers=True):
        super().__init__(normalized_shape)

        self.register_buffer('op_scale', 0.1*torch.ones(1), persistent=persistant_buffers)
        self.register_buffer('op_scale_bw', 0.1*torch.ones(1), persistent=persistant_buffers)

    def forward(self, x):
        x = gradient_observer.apply(x, self.op_scale_bw)
        x = super().forward(x)
        self.op_scale = torch.atleast_1d(torch.tensor(tensor_scale(x.detach())))
        return x


class GroupNorm(GN):
    def __init__(self, num_groups, num_channels, persistant_buffers=True):
        super().__init__(num_groups, num_channels)

        self.register_buffer('op_scale', 0.1*torch.ones(1), persistent=persistant_buffers)
        self.register_buffer('op_scale_bw', 0.1*torch.ones(1), persistent=persistant_buffers)

    def forward(self, x):
        x = gradient_observer.apply(x, self.op_scale_bw)
        x = super().forward(x)
        self.op_scale = torch.atleast_1d(torch.tensor(tensor_scale(x.detach())))
        return x


class Linear(Lnr):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, persistant_buffers=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('op_scale', 0.1*torch.ones(1), persistent=persistant_buffers)
        self.register_buffer('op_scale_bw', 0.1*torch.ones(1), persistent=persistant_buffers)

    def forward(self, x):
        x = gradient_observer.apply(x, self.op_scale_bw)
        x = super().forward(x)
        self.op_scale = torch.atleast_1d(torch.tensor(tensor_scale(x.detach())))
        return x


class Conv2d(CV2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=False, persistant_buffers=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                            groups=groups, bias=bias)

        self.register_buffer('op_scale', 0.1*torch.ones(1), persistent=persistant_buffers)
        self.register_buffer('op_scale_bw', 0.1*torch.ones(1), persistent=persistant_buffers)

    def forward(self, x):
        x = gradient_observer.apply(x, self.op_scale_bw)
        x = super().forward(x)
        self.op_scale = torch.atleast_1d(torch.tensor(tensor_scale(x.detach())))
        return x


class Add(torch.nn.Module):
    def __init__(self, persistant_buffers=True):
        super().__init__()
        self.register_buffer('op_scale', 0.1*torch.ones(1), persistent=persistant_buffers)

    def forward(self, x1, x2):
        out = x1 + x2
        self.op_scale = torch.atleast_1d(torch.tensor(tensor_scale(out.detach())))
        return out


class Cat(torch.nn.Module):
    def __init__(self, persistant_buffers=True):
        super().__init__()
        self.register_buffer('op_scale', 0.1*torch.ones(1), persistent=persistant_buffers)

    def forward(self, list_of_tensors, dim=0):
        out = torch.cat(list_of_tensors, dim=dim)
        self.op_scale = torch.atleast_1d(torch.tensor(tensor_scale(out.detach())))
        return out


class gradient_observer(Function):
    @staticmethod
    def forward(ctx, input, grad_scale_bw):
        ctx.set_materialize_grads(False)
        ctx.grad_scale_bw = grad_scale_bw
        return input

    @staticmethod
    def backward(ctx, grad_output):
        scale = torch.atleast_1d(torch.tensor(tensor_scale(grad_output)))
        ctx.grad_scale_bw.set_(scale.to(ctx.grad_scale_bw.device))
        return grad_output, None