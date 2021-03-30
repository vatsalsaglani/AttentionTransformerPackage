import torch 
import torch.nn as nn
from .EncoderLayer import EncoderLayer
from .PositionalEncoding import PositionalEncoding


class Encoder(nn.Module):

    def __init__(
        self, source_vocab_size, emb_dim, layers, heads, dim_key,
        dim_value, dim_model, dim_inner, pad_id, dropout = 0.1, num_pos = 200
        ):

        super().__init__()

        self.word_embedding = nn.Embedding(source_vocab_size, emb_dim, padding_idx = pad_id)
        self.position_encoding = PositionalEncoding(emb_dim, num_pos = num_pos)

        self.dropout = nn.Dropout(p = dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(
                dim_model, dim_inner, heads, dim_key, dim_value, dropout = dropout
            ) for _ in range(layers)
        ])

        self.layer_norm = nn.LayerNorm(dim_model, eps = 1e-6)

    def forward(self, source_seq, source_mask, return_attentions = False):

        encoder_self_attention_list = []

        

        encoder_output = self.dropout(self.position_encoding(self.word_embedding(source_seq.long())))

        encoder_output = encoder_output.float()

        for encoder_layer in self.layer_stack:

            encoder_output, encoder_self_attention = encoder_layer(encoder_output, self_attention_mask = source_mask)

            encoder_self_attention_list += [encoder_self_attention] if return_attentions else []

        encoder_output = self.layer_norm(encoder_output)

        if return_attentions:

            return encoder_output, encoder_self_attention_list

        return encoder_output
