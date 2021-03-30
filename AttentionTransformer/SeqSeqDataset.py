import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class SeqSeqDataset(Dataset):
    """
    Dataset object to create batches using a DataLoader
    args:
    tokens: A list of lists wherein every list contains input tokens
    labels: A list of lists where each list contains labels corresponding to the input list
    seq_len: Choise of sequence length 
    pbit: padding bit for the target
    Input padding bit is assumed to be '0'
    """

    def __init__(self, tokens: list, labels: list, seq_len: int, pbit: int):

        self.tokens = tokens
        self.labels = labels
        self.seq_len = seq_len
        self.pbit = pbit

    def pad_sequences(self, arr, islabel = False):

        if len(arr) > self.seq_len:
            arr = arr[:self.seq_len]

        arr = torch.tensor(arr)

        op = torch.zeros((self.seq_len))
        if islabel:
            
            op[op == 0] = self.pbit
        
        op[:arr.size(0)] = arr

        return op.float()

    
    def __len__(self):

        return len(self.tokens)

    
    def __getitem__(self, ix):

        token = self.tokens[ix]
        label = self.labels[ix]

        source_tnsr = self.pad_sequences(token)
        target_tnsr = self.pad_sequences(label, islabel=True)

        return {'src': source_tnsr, 'trg': target_tnsr}


