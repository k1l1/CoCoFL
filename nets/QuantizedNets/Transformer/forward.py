import torch
import torch.nn as nn
from nets.QuantizedNets.Transformer.training import MultiHeadAttention
from nets.QuantizedNets.utils.forward import QLayerNorm, QLinear, QAdd, QLinearReLU
from nets.QuantizedNets.utils.utils import tensor_scale, filter_state_dict_keys


class QFWEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff, is_transition=False):
        super().__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.linear_q = QLinear(d_model, d_model)
        self.linear_k = QLinear(d_model, d_model)
        self.linear_v = QLinear(d_model, d_model)

        self.mha = MultiHeadAttention(d_model, n_heads)

        self.linear_o = QLinear(n_heads * d_model//n_heads, d_model)

        self.layernorm1 = QLayerNorm(d_model)

        self.linear1_relu = QLinearReLU(d_model, d_ff)
        self.linear2 = QLinear(d_ff, d_model)

        self.layernorm2 = QLayerNorm(d_model)

        self.qadd1 = QAdd()
        self.qadd2 = QAdd()

        self.is_transition = is_transition

    def forward(self, inputs, attn_mask):
        if not inputs.is_quantized:
            inputs = torch.quantize_per_tensor(inputs, tensor_scale(inputs), 64, dtype=torch.quint8)

        q_heads = self.linear_q(inputs)
        k_heads = self.linear_v(inputs)
        v_heads = self.linear_k(inputs)

        attn_outputs = self.mha(torch.dequantize(q_heads), torch.dequantize(k_heads),
                                    torch.dequantize(v_heads), attn_mask)

        attn_outputs = torch.quantize_per_tensor(attn_outputs, tensor_scale(attn_outputs), 64, dtype=torch.quint8)

        attn_outputs = self.linear_o(attn_outputs)
        attn_outputs = self.qadd1(inputs, attn_outputs)
        attn_outputs = self.layernorm1(attn_outputs)
        ffn_outputs = self.linear1_relu(attn_outputs)
        ffn_outputs = self.linear2(ffn_outputs)

        ffn_outputs = self.qadd2(attn_outputs, ffn_outputs)
        ffn_outputs = self.layernorm2(ffn_outputs)

        if self.is_transition:
            return torch.dequantize(ffn_outputs)
        else:
            return ffn_outputs

    def sd_hook(self, state_dict, prefix, *_):
        self.linear_q.init(filter_state_dict_keys(state_dict, prefix + 'linear_q.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_q.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_q.op_scale'))

        self.linear_v.init(filter_state_dict_keys(state_dict, prefix + 'linear_v.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_v.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_v.op_scale'))

        self.linear_k.init(filter_state_dict_keys(state_dict, prefix + 'linear_k.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_k.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_k.op_scale'))

        self.linear_o.init(filter_state_dict_keys(state_dict, prefix + 'linear_o.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_o.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_o.op_scale'))

        self.linear1_relu.init(filter_state_dict_keys(state_dict, prefix + 'linear1.weight'),
                               filter_state_dict_keys(state_dict, prefix + 'linear1.bias'),
                               filter_state_dict_keys(state_dict, prefix + 'linear1.op_scale'))

        self.linear2.init(filter_state_dict_keys(state_dict, prefix + 'linear2.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear2.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear2.op_scale'))

        self.layernorm1.init(filter_state_dict_keys(state_dict, prefix + 'layernorm1.weight'),
                             filter_state_dict_keys(state_dict, prefix + 'layernorm1.bias'),
                             filter_state_dict_keys(state_dict, prefix + 'layernorm1.op_scale'))

        self.layernorm2.init(filter_state_dict_keys(state_dict, prefix + 'layernorm2.weight'),
                             filter_state_dict_keys(state_dict, prefix + 'layernorm2.bias'),
                             filter_state_dict_keys(state_dict, prefix + 'layernorm2.op_scale'))

        self.qadd1.init(filter_state_dict_keys(state_dict, prefix + 'add1.op_scale'))
        self.qadd2.init(filter_state_dict_keys(state_dict, prefix + 'add2.op_scale'))
