import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):

    def __init__(self, data_array, seq_length):

        self.data_array = data_array
        self.seq_length = seq_length
        self.total_words = len(self.data_array)

        self.req_size = self.total_words - self.seq_length - 1

    def __len__(self):

        return self.req_size

    def __getitem__(self, ix):

        inp_seq = torch.FloatTensor(self.data_array[ix:ix+self.seq_length])
        op_seq = torch.FloatTensor(self.data_array[ix+1:ix+self.seq_length+1])

        return {'src': inp_seq, 'trg': op_seq}

        