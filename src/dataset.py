"""
Stage C: wraps the processed feature table in a PyTorch Dataset.
"""
from torch.utils.data import Dataset

class DefensiveShellDataset(Dataset):
    def __init__(self, features_df, labels):
        # TODO: store tensors
        pass

    def __len__(self):
        # TODO
        pass

    def __getitem__(self, idx):
        # TODO: return (features, label) as tensors
        pass