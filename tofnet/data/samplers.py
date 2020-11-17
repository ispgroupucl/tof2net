from torch.utils.data.sampler import *

class RandomNSampler(Sampler):
    def __init__(self, data_source, n=4):
        self.data_source = data_source
        self.n = n

    @property
    def num_samples(self):
        # dataset size might change at runtime
        return len(self.data_source)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n = len(self.data_source) // self.n
        for i in torch.randperm(n).tolist():
            for j in range(self.n):
                if i+j < self.num_samples:
                    yield i+j

def Random4Sampler(data_source):
    return RandomNSampler(data_source, n=4)
