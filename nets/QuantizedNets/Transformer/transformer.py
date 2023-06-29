from nets.QuantizedNets.Transformer.backward import QBWEncoderLayer
from nets.QuantizedNets.Transformer.training import EncoderLayer
from nets.QuantizedNets.Transformer.forward import QFWEncoderLayer
import numpy as np
import torch
import torch.nn as nn
import random
import json

from nets.QuantizedNets.utils.utils import filter_table

with open("nets/QuantizedNets/Transformer/tables/table__CoCoFL_x64_QTransformer.json", "r") as fd:
    _g_table_qtransformer = json.load(fd)

with open("nets/QuantizedNets/Transformer/tables/table__CoCoFL_x64_QTransformerSeq2Seq.json", "r") as fd:
    _g_table_qtransformerseq2seq = json.load(fd)


class QTransformerBase(nn.Module):
    def __init__(self, trained_block_list, fw_block_list, bw_block_list, freeze_idxs=[],
                 vocab_size=16000, seq_len=1024, d_model=512, n_layers=6, n_heads=8, p_drop=0.1, d_ff=2048, pad_id=0):
        super(QTransformerBase, self).__init__()

        # CoCoFL related
        layer_idxs = [i for i in range(1 + 6 + 1)]
        self.max_idxs = 1 + 6
        assert set(freeze_idxs) <= set(layer_idxs), "Invalid layer idxs"

        self._trained_idxs = [idx for idx in layer_idxs if idx not in freeze_idxs]
        assert set(self._trained_idxs) == set([i for i in range(max(self._trained_idxs) + 1 - len(self._trained_idxs), max(self._trained_idxs) + 1)]), \
             "No continous block of trained layers"

        self._trained_encoder_block = trained_block_list
        self._fw_encoder = fw_block_list
        self._bw_encoder = bw_block_list

        # Transformer related
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(seq_len+1, d_model)  # (seq_len+1, d_model)

        # Layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)

        if 0 not in self._trained_idxs:
            for parameter in self.embedding.parameters():
                parameter.requires_grad = False
            self.embedding.eval()
            for parameter in self.pos_embedding.parameters():
                parameter.requires_grad = False
            self.pos_embedding.eval()

        self._block_idx = 1

        list_of_encoders = []
        for _ in range(n_layers):

            # Case fw layer(frozen)
            if self._block_idx < min(self._trained_idxs):
                transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
                list_of_encoders.append(self._fw_encoder(d_model, n_heads,
                                        p_drop, d_ff, is_transition=transition))

            # Case bw+(fw) layer (also frozen)
            elif self._block_idx > max(self._trained_idxs):
                list_of_encoders.append(self._bw_encoder(d_model, n_heads,
                                        p_drop, d_ff))

            # Case trained layer
            elif self._block_idx in self._trained_idxs:
                is_first = True if self._block_idx == min(self._trained_idxs) else False
                list_of_encoders.append(self._trained_encoder_block(d_model, n_heads,
                                        p_drop, d_ff, is_first=is_first))

            self._block_idx += 1

        self.layers = nn.ModuleList(list_of_encoders)
        # layers to classify
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.Softmax(dim=-1)

        if self.max_idxs not in self._trained_idxs:
            for parameter in self.linear.parameters():
                parameter.requires_grad = False

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




class QTransformer(QTransformerBase):
    def __init__(self, freeze=[]):

        super().__init__(EncoderLayer, QFWEncoderLayer, QBWEncoderLayer, freeze_idxs=freeze,
                         vocab_size=16000, seq_len=512, d_model=128, n_layers=6,
                         n_heads=2, p_drop=0.05, d_ff=128, pad_id=0)

    @staticmethod
    def n_freezable_layers():
        return 1 + 6 + 1

    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qtransformer, QTransformer.n_freezable_layers())
        return random.choice(configs)


class QTransformerSeq2Seq(QTransformerBase):
    def __init__(self, freeze=[]):
        d_model = 128
        n_classes = 81
        super().__init__(EncoderLayer, QFWEncoderLayer, QBWEncoderLayer, freeze_idxs=freeze,
                         vocab_size=81, seq_len=81, d_model=d_model, n_layers=6,
                         n_heads=2, p_drop=0.05, d_ff=128, pad_id=80)

        self.linear = nn.Linear(d_model, n_classes)

        if self.max_idxs not in self._trained_idxs:
            for parameter in self.linear.parameters():
                parameter.requires_grad = False

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)

        for layer in self.layers:
            outputs = layer(outputs, attn_pad_mask)

        outputs = self.linear(outputs)

        outputs = outputs.permute(0, 2, 1)

        return outputs[:, -1, :]

    @staticmethod
    def n_freezable_layers():
        return 1 + 6 + 1

    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qtransformerseq2seq, QTransformerSeq2Seq.n_freezable_layers())
        return random.choice(configs)