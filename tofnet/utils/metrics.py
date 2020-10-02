import numpy as np
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import *
from ignite.contrib.metrics import *


class Metrics(dict):
    def __init__(self, **kwargs):
        self.hidden = {}
        super().__init__(**kwargs)

    def add_hidden(self, key, value):
        self.hidden[key] = value

class MetricList(Metric):
    def __init__(self, metrics=None, show=None, ignore_index=255):
        self.metrics = metrics or {}
        self.show = show or []
        self.ignore_index = ignore_index

    def __setitem__(self, key, metric):
        self.metrics[key] = metric

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def ignore(self, y):
        if type(y) != list:
            y = [y]
        count_ignore = 0
        for yy in y:
            count_ignore += torch.eq(yy, self.ignore_index).all()
        return count_ignore == len(y)

    @torch.no_grad()
    def update(self, output):
        for metric in self.metrics.values():
            _, y = transformed_output = metric._output_transform(output)
            if self.ignore(y):
                continue
            metric.update(transformed_output)

    def compute(self):
        outputs = Metrics()
        for name, metric in self.metrics.items():
            if name.rsplit("_",1)[-1] in self.show:
                try:
                    outputs[name] = metric.compute().data
                except NotComputableError:
                    outputs[name] = float("NaN")
            else:
                try:
                    outputs.add_hidden(name, metric.compute())
                except NotComputableError:
                    outputs.add_hidden(name, float("NaN"))
        return outputs

class MSEWithLogits(Metric):
    """
    Calculates the mean squared error with sigmoid activation.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_squared_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        squared_errors = torch.pow(torch.sigmoid(y_pred) - y.view_as(y_pred), 2)
        self._sum_of_squared_errors += torch.mean(squared_errors).item() * y.shape[0]
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanSquaredError must have at least one example before it can be computed.')
        return self._sum_of_squared_errors / self._num_examples

class MeanIoU(Metric):
    def __init__(self, cm, ignore_index=None, output_transform=lambda x:x):
        self.cm = cm # ConfusionMatrix(n_classes, output_transform=output_transform)
        self.ignore_index = ignore_index
        self.iou = mIoU(self.cm, ignore_index=self.ignore_index)
        super().__init__(output_transform)

    def reset(self):
        self.cm.reset()

    def update(self, output):
        if self.ignore_index:
            output[1][output[1]==self.ignore_index] = 0
        self.cm.update(output)

    def compute(self):
        return self.iou.compute().item()

class ClassIoU(Metric):
    def __init__(self, cm, choose_index=None, output_transform=lambda x:x):
        self.cm = cm # ConfusionMatrix(n_classes, output_transform=output_transform)
        self.choose_index = choose_index
        self.iou = IoU(self.cm, ignore_index=None)
        super().__init__(output_transform)

    def reset(self):
        self.cm.reset()

    def update(self, output):
        self.cm.update(output)

    def compute(self):
        if self.choose_index is None:
            return [x.item() for x in self.iou.compute()]
        else:
            return self.iou.compute()[self.choose_index].item()