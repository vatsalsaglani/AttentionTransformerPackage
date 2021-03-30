import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset 

class ClassificationDataset(Dataset):

    def __init__(self, texts: list, labels: list, seq_len: int):


        self.labels = labels
        self.texts = texts
        self.seq_len = seq_len


    def pad_sequences(self, arr):

        if len(arr) > self.seq_len:

            arr = arr[:self.seq_len]

        arr = torch.tensor(arr)

        op = torch.zeros((self.seq_len))

        op[:arr.size(0)] = arr

        return op

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, ix):

        label = self.labels[ix]
        text = self.texts[ix]
        text = self.pad_sequences(text)

        return {"src": text.float(), "label": torch.tensor(label)}


