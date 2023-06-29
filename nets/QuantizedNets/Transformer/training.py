import torch
import torch.nn as nn
import numpy as np
from nets.QuantizedNets.utils.training import Linear, LayerNorm, Add


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attn_mask):
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_score.masked_fill_(attn_mask, -1e9)

        attn_weights = nn.Softmax(dim=-1)(attn_score)

        output = torch.matmul(attn_weights, v)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model//n_heads

        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        q_heads = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        attn = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)

        return attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff, is_first=False):
        super(EncoderLayer, self).__init__()

        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)

        self.mha = MultiHeadAttention(d_model, n_heads)

        self.linear_o = Linear(n_heads * d_model//n_heads, d_model)

        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = LayerNorm(d_model)

        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu = nn.ReLU()

        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = LayerNorm(d_model)

        self.add1 = Add()
        self.add2 = Add()

    def forward(self, inputs, attn_mask):
        q_heads = self.linear_q(inputs)
        k_heads = self.linear_v(inputs)
        v_heads = self.linear_k(inputs)

        attn_outputs = self.mha(q_heads, k_heads, v_heads, attn_mask)

        attn_outputs = self.linear_o(attn_outputs)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.add1(inputs, attn_outputs)

        attn_outputs = self.layernorm1(attn_outputs)

        ffn_outputs = self.linear1(attn_outputs)
        ffn_outputs = self.relu(ffn_outputs)
        ffn_outputs = self.linear2(ffn_outputs)

        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.add2(attn_outputs, ffn_outputs)
        ffn_outputs = self.layernorm2(ffn_outputs)

        return ffn_outputs
