import torch.nn as nn
import math


class FreezableBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        """ Easier to freeze batchnorm2d layer
        """
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.frozen = False

    def train(self, mode=True):
        if self.frozen:
            super().train(False)
        else:
            super().train(mode)


class ModelConfig():
    ''' Contains the layers used for the model (fixed for now)
    '''
    def __init__(self, bn=None, act=None, weight=None):
        self.conv = nn.Conv2d
        self.bn = FreezableBatchNorm2d
        self.act = nn.LeakyReLU
        self.fc = nn.Linear
        self.conv_t = nn.ConvTranspose2d


def freeze_module(module):
    cnt = 0
    for name, weights in module.named_parameters():
        print(name)
        cnt += 1
        weights.requires_grad = False
    print(f"froze approx {cnt/2} layers")
    for name, m in module.named_modules():
        if isinstance(m, FreezableBatchNorm2d):
                m.frozen = True
        # elif m != module: # avoid recursive calls on itself
        #     freeze_module(m)
