from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader

class IDDataset(Dataset):
    def __init__(self, total) -> None:
        super().__init__()

        self.total = total

    def __getitem__(self, index: Any) -> Any:
        return index

    def __len__(self):
        return self.total

def get_simple_sampler(total, batch):
    id_dataset = IDDataset(total)
    sampler = DataLoader(id_dataset, batch, shuffle=True)

    return sampler

if __name__ == '__main__':
    sampler = get_simple_sampler(100, 10)

    outlist = []
    for i in sampler:
        outlist.append(i)
    pass

    outlist = []
    for i in range(200):
        outp = next(iter(sampler))
        print(i)
        if i < 10:
            outlist.append(outp)
        pass
