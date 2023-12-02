from itertools import product

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class ProductDataset(Dataset):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.datasets = args[0]
        else:
            self.datasets = list(args)

        lens = map(lambda ds: len(ds), self.datasets)
        self.combine = list(product(*map(lambda length: range(length), lens)))

    def __getitem__(self, index) -> T_co:
        combine = self.combine[index]
        return tuple(map(lambda ds, c: ds[c], self.datasets, combine))

    def __len__(self):
        return len(self.combine)
