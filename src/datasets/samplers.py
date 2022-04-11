import torch
from torch.utils.data import Sampler
import numpy as np




class RandomSubsetSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    Arguments:
        data_source (Dataset): dataset to sample from
        subset_rate (float): rate of subset to whole dataset
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
    """

    def __init__(self, data_source, subset_rate=1.0, replacement=False):
        self.data_source = data_source
        self.subset_rate = subset_rate
        self.replacement = replacement

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        return int(len(self.data_source) * self.subset_rate)

    def __iter__(self):
        n = len(self.data_source)

        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist()[:self.num_samples])

    def __len__(self):
        return self.num_samples


class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, min_size=600, max_height=800, max_width=800, size_int=8):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.hmin = min_size
        self.hmax = max_height
        self.wmin = min_size
        self.wmax = max_width
        self.size_int = size_int
        self.hint = (self.hmax-self.hmin)//self.size_int+1
        self.wint = (self.wmax-self.wmin)//self.size_int+1

    def generate_height_width(self):
        hi, wi = np.random.randint(0, self.hint), np.random.randint(0, self.wint)
        h, w = self.hmin + hi * self.size_int, self.wmin + wi * self.size_int
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

