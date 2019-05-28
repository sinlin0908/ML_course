import torch
from torch.utils.data import Dataset
import numpy as np


class StockDataset(Dataset):
    def __init__(self, data):
        x, y = data
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(np.array(y)).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
