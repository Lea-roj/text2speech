import bisect
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MNISTAudio(Dataset):
    def __init__(self, path="wavenet_dataset.npz", src_len=1088, tgt_len=64, num_class=256):
        self.path = path
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.num_class = num_class
        self.dataset_index = self.index_dataset()

    def index_dataset(self):
        dataset_index = [0]
        dataset = np.load(self.path, mmap_mode='r')
        for i in range(len(dataset.files)):
            dataset_index.append(dataset_index[-1] + dataset[f'arr_{i}'].shape[0])
        return dataset_index

    def __getitem__(self, idx):
        d_idx = bisect.bisect_right(self.dataset_index, idx)
        dataset = np.load(self.path, mmap_mode='r')

        if idx + self.src_len + 1 <= self.dataset_index[d_idx]:
            start_pos = idx - self.dataset_index[d_idx - 1]
            end_pos = start_pos + self.src_len + 1
            data = dataset[f'arr_{d_idx - 1}'][start_pos:end_pos]
        else:
            start_pos = idx - self.dataset_index[d_idx - 1]
            end_pos = idx + self.src_len + 1 - self.dataset_index[d_idx]
            data = np.concatenate((
                dataset[f'arr_{d_idx - 1}'][start_pos:],
                dataset[f'arr_{d_idx}'][:end_pos]
            ))

        src = torch.tensor(data[:self.src_len])
        src = F.one_hot(src, self.num_class).type(torch.float).transpose(0, 1)
        tgt = torch.tensor(data[-self.tgt_len:])
        return {"src": src, "tgt": tgt}

    def __len__(self):
        return self.dataset_index[-1] - self.src_len