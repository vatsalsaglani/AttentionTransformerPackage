import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoder import Encoder
from .Decoder import Decoder

class Transformer(nn.Module):

    def __init__(
        self, source_vocab_size, target_vocab_size, source_pad_id, target_pad_id, emb_dim = 512, dim_model = 512, 
        dim_inner = 2048, layers = 6, heads = 8, dim_key = 64, dim_value = 64, dropout = 0.1, num_pos = 200,
        target_emb_projection_weight_sharing = True, emb_source_target_weight_sharing = True
        ):

        super(Transformer, self).__init__()

        self.source_pad_id, self.target_pad_id = source_pad_id, target_pad_id

        self.encoder = Encoder(
            source_vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, source_pad_id, dropout = dropout, num_pos = num_pos
        )

        self.decoder = Decoder(
            target_vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, target_pad_id, dropout = dropout, num_pos = num_pos
        )

        self.target_word_projection = nn.Linear(dim_model, target_vocab_size, bias = False)

        for parameter in self.parameters():

            if parameter.dim() > 1:

                nn.init.xavier_uniform_(parameter)

        assert dim_model == emb_dim, f'Dimensions of all the module outputs must be same'

        self.x_logit_scale = 1

        if target_emb_projection_weight_sharing:

            self.target_word_projection.weight = self.decoder.word_embedding.weight

            self.x_logit_scale = (dim_model ** -0.5)

        if emb_source_target_weight_sharing:

            self.encoder.word_embedding.weight = self.decoder.word_embedding.weight

    
    def get_pad_mask(self, sequence, pad_id):

        return (sequence != pad_id).unsqueeze(-2)

    def get_subsequent_mask(self, sequence):

        batch_size, seq_length = sequence.size()

        subsequent_mask = (
            1 - torch.triu(
                torch.ones((1, seq_length, seq_length), device=sequence.device), diagonal = 1
            )
        ).bool()

        return subsequent_mask

    def forward(self, source_seq, target_seq):

        source_mask = self.get_pad_mask(source_seq, self.source_pad_id)
        target_mask = self.get_pad_mask(target_seq, self.target_pad_id) & self.get_subsequent_mask(target_seq)

        encoder_output = self.encoder(source_seq, source_mask)
        decoder_ouptut = self.decoder(target_seq, target_mask, encoder_output, source_mask)

        seq_logit = self.target_word_projection(decoder_ouptut) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))