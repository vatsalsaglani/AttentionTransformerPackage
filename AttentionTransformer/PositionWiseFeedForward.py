import torch 
import torch.nn as nn
import torch.nn.functional as F
from .Activations import activation_dict


class PositionWiseFeedForward(nn.Module):

    '''
    Feed Forward Layer
    '''

    def __init__(self, dim_in, dim_hid, elu_func: str = 'gelu', dropout = 0.1):

        super().__init__()

        self.feedforward_1 = nn.Linear(dim_in, dim_hid)
        self.feedforward_2 = nn.Linear(dim_hid, dim_in)

        self.layer_norm = nn.LayerNorm(dim_in, eps = 1e-6)
        self.dropout = nn.Dropout(dropout)
        self.elu = activation_dict[elu_func]


    def forward(self, inp):

        residual = inp
        inp = self.layer_norm(inp)

        inp = self.feedforward_2(self.elu(self.feedforward_1(inp)))

        inp = self.dropout(inp)

        inp += residual

        return inp
