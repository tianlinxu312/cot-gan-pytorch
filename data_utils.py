import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


class MyDataset(IterableDataset):

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

