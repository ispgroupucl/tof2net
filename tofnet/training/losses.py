import numpy as np
import operator
from itertools import cycle
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.loss import *
from kornia.losses import *

class Loss:
    def compute(self, output, target):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def __add__(self, other):
        return LossesLambda(operator.add, self, other)

    def __radd__(self, other):
        return LossesLambda(operator.add, other, self)

    def __sub__(self, other):
        return LossesLambda(operator.sub, self, other)

    def __rsub__(self, other):
        return LossesLambda(operator.sub, other, self)

    def __mul__(self, other):
        return LossesLambda(operator.mul, self, other)

    def __rmul__(self, other):
        return LossesLambda(operator.mul, other, self)

class LossesLambda(Loss):
    def __init__(self, f, *args, **kwargs):
        self.function = f
        self.args = args
        self.kwargs = kwargs

    def compute(self, output, target):
        materialized = [i.compute(output, target) if isinstance(i, Loss) else i for i in self.args]
        materialized_kwargs = {k: (v.compute(output, target) if isinstance(v, Loss) else v) for k, v in self.kwargs.items()}
        return self.function(*materialized, **materialized_kwargs)


class Criterion(Loss):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def compute(self, output, target):
        return self.loss_fn(output, target)

class MultiCriterion:
    def __init__(self, losses:[Loss], weights=None):
        self.losses = losses
        self.weights = weights or [1] * len(losses)

    def compute(self, outputs, targets):
        output_loss = 0.0
        for output, target, loss, weight in zip(outputs, targets, cycle(self.losses), cycle(self.weights)):
            output_loss += weight * loss(output, target)
        return output_loss

    def __getitem__(self, idx):
        return self.losses[idx % len(self.losses)]

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


"""
    ####################
      Single-class CNN
    ####################
"""
class IntBCEWithLogitsLoss(BCEWithLogitsLoss):
    def ___init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return super().forward(input, target.float())
