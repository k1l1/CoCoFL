import torch
from nets.QuantizedNets.Transformer.training import EncoderLayer, MultiHeadAttention
import torch.nn as nn
from nets.QuantizedNets.utils.backwards import QBWLinear
from nets.QuantizedNets.utils.utils import filter_state_dict_keys


class QBWEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff, is_first=False):
        super(QBWEncoderLayer, self).__init__()
        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.linear_q = QBWLinear(d_model, d_model)
        self.linear_k = QBWLinear(d_model, d_model)
        self.linear_v = QBWLinear(d_model, d_model)

        self.mha = MultiHeadAttention(d_model, n_heads)

        self.linear_o = QBWLinear(n_heads * d_model//n_heads, d_model)

        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model)

        self.linear1 = QBWLinear(d_model, d_ff)
        self.linear2 = QBWLinear(d_ff, d_model)
        self.relu = nn.ReLU()

        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, inputs, attn_mask):
        q_heads = self.linear_q(inputs)
        k_heads = self.linear_v(inputs)
        v_heads = self.linear_k(inputs)

        attn_outputs = self.mha(q_heads, k_heads, v_heads, attn_mask)

        attn_outputs = self.linear_o(attn_outputs)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = inputs + attn_outputs

        attn_outputs = self.layernorm1(attn_outputs)

        ffn_outputs = self.linear1(attn_outputs)
        ffn_outputs = self.relu(ffn_outputs)
        ffn_outputs = self.linear2(ffn_outputs)

        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = attn_outputs + ffn_outputs
        ffn_outputs = self.layernorm2(ffn_outputs)

        return ffn_outputs

    def sd_hook(self, state_dict, prefix, *_):
        self.linear_q.init(filter_state_dict_keys(state_dict, prefix + 'linear_q.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_q.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_q.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_q.op_scale_bw'))

        self.linear_v.init(filter_state_dict_keys(state_dict, prefix + 'linear_v.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_v.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_v.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_v.op_scale_bw'))

        self.linear_k.init(filter_state_dict_keys(state_dict, prefix + 'linear_k.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_k.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_k.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_k.op_scale_bw'))

        self.linear_o.init(filter_state_dict_keys(state_dict, prefix + 'linear_o.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_o.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_o.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'linear_o.op_scale_bw'))

        self.linear1.init(filter_state_dict_keys(state_dict, prefix + 'linear1.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear1.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear1.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'linear1.op_scale_bw'))

        self.linear2.init(filter_state_dict_keys(state_dict, prefix + 'linear2.weight'),
                            filter_state_dict_keys(state_dict, prefix + 'linear2.bias'),
                            filter_state_dict_keys(state_dict, prefix + 'linear2.op_scale'),
                            filter_state_dict_keys(state_dict, prefix + 'linear2.op_scale_bw'))
