import numpy as np
import torch 
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, dim_hid, num_pos):

        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(num_pos, dim_hid))

    def _get_sinusoid_encoding_table(self, num_pos, dim_hid):

        def get_position_angle_vec(position):

            return [
                position / np.power(10000, 2 * (hid_j // 2) / dim_hid) for hid_j in range(dim_hid)
            ]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_pos)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()