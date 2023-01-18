from torch.utils.data import Dataset
import torch

# Pytorch dataset for tabular data (x and y are simply 2D tensors)
class TabularDataset(Dataset):
    def __init__(self, labelled_data):
        super(TabularDataset, self).__init__()
        self.ys = labelled_data.y

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, idx):
        y_row = self.ys[idx]

        return y_row

