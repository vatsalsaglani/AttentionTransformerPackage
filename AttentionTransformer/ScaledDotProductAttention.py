import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):

    ''' Scaled Attention ''' 

    def __init__(self, temperature, attention_dropout = 0.1):

        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, mask = None):

        attention = torch.matmul(query / self.temperature, key.transpose(2, 3))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = self.dropout(torch.softmax(attention, dim = -1))

        output = torch.matmul(attention, value)

        return output, attention


        