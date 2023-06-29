import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.mha = MultiHeadAttention(d_model, n_heads)

        self.linear_o = nn.Linear(n_heads * d_model//n_heads, d_model)

        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        q_heads = self.linear_q(inputs)
        k_heads = self.linear_v(inputs)
        v_heads = self.linear_k(inputs)

        attn_outputs = self.mha(q_heads, k_heads, v_heads, attn_mask)

        attn_outputs = self.linear_o(attn_outputs)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        ffn_outputs = self.linear1(attn_outputs)
        ffn_outputs = self.relu(ffn_outputs)
        ffn_outputs = self.linear2(ffn_outputs)

        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)

        return ffn_outputs


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size=16000, seq_len=1024, d_model=512, n_layers=6, n_heads=8, p_drop=0.05, d_ff=2048, pad_id=0, n_classes=2):
        super(TransformerEncoder, self).__init__()
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(seq_len+1, d_model)  # (seq_len+1, d_model)

        # layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])
        # layers to classify
        self.linear = nn.Linear(d_model, n_classes)
        self.softmax = nn.Softmax(dim=-1)

        self.linear.weight.data.uniform_(-1e-5, 1e-5)
        self.linear.bias.data.uniform_(-1e-5, 1e-5)

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)
        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)

        for layer in self.layers:
            outputs = layer(outputs, attn_pad_mask)

        outputs, _ = torch.max(outputs, dim=1)
        outputs = self.softmax(self.linear(outputs))

        return outputs

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)

        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i %2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)


class Transformer(TransformerEncoder):
    def __init__(self):
        super().__init__(vocab_size=16000, seq_len=512, d_model=128, n_layers=6,
                         n_heads=2, p_drop=0.05, d_ff=128, pad_id=0)


class TransformerSeq2Seq(TransformerEncoder):
    def __init__(self):
        super().__init__(vocab_size=81, seq_len=81, d_model=128, n_layers=6,
                         n_heads=2, p_drop=0.05, d_ff=128, pad_id=80, n_classes=81)

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)

        for layer in self.layers:
            outputs = layer(outputs, attn_pad_mask)
        # For Seq2Seq, no max of embeddings
        outputs = self.linear(outputs)

        # Permute output to batch, position, probs
        outputs = outputs.permute(0, 2, 1)

        # Return only estimation of last char
        return outputs[:, -1, :]