import torch
from torch.utils.data import Dataset

class FraudDataset(Dataset):
    def __init__(self, X):
        # convertir X en tensor PyTorch
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]