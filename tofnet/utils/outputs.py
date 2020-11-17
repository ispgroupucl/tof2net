import torch
from abc import ABCMeta
from tofnet import utils

class Output(metaclass=ABCMeta):
    """ Object representing the output type of a neural network.
        This class should be extended for every output type in order to add
        the desired metrics computed on the output.

    Arguments:
        i (int): index of the output
        shape (tuple): shape of the feature map
        n_outputs (int): number of output feature maps
        postfix (str): suffix in order to differentiate identical outputs
        type_suffix (str): suffix that adds information about the output-type
    """
    def __init__(self, i, shape, n_outputs, postfix="", type_suffix=""):
        self.i = i
        self.shape = shape
        self.n_outputs = n_outputs
        self.postfix = postfix
        self.metrics = None
        self.output_transform = lambda x, i=i: [x[0][i], x[1][i]]
        self.suffix = f"_{type_suffix}" if type_suffix != "" else type_suffix

    @property
    def name(self):
        return type(self).__name__.lower()

    @property
    def full_name(self):
        return f"{type(self).__name__.lower()}{self.suffix}"

    def reset(self):
        for metric in self.metrics["update"]:
            metric.reset()

    @torch.no_grad()
    def update(self, output):
        for metric in self.metrics["update"]:
            metric.update(output)

    def summary(self):
        return self._compute("summary")

    def compute(self):
        return self._compute("scalars")

    def _compute(self, key):
        outputs = {}
        for name, metric in self.metrics[key].items():
            try:
                outputs[name] = metric.compute()
                if hasattr(outputs[name], "item"):
                    outputs[name] = outputs[name].item()
            except utils.metrics.NotComputableError:
                outputs[name] = float("NaN")
        return outputs


    def create_metrics(self, criterion, ignore_index):
        metrics = {}
        metrics["scalars"] = {f"{self.name}_acc": utils.metrics.Accuracy(output_transform=self.output_transform),
                f"{self.name}_loss": utils.metrics.Loss(criterion, output_transform=self.output_transform)}
        metrics["summary"] = {} # metrics["scalars"]
        metrics["update"] = list(metrics["scalars"].values())
        metrics["matrices"] = {}
        self.metrics = metrics

    def __str__(self):
        shape = "x".join([str(x) for x in self.shape])
        return f"{self.name}_{shape}" + (f"_{self.postfix}" if self.postfix else "")

    def __hash__(self):
        return hash(self.full_name)

    def __eq__(self, other):
        if type(other) == str:
            return self.full_name == other
        elif type(other) == Output:
            return self.full_name == other.full_name
        else:
            return False

class Keypoints(Output):
    """ Handles the Keypoint regression output type. Adds the MSE loss computation """
    def __init__(self, i, shape, n_outputs, type_suffix=""):
        super().__init__(i, shape, n_outputs, type_suffix=type_suffix)

    def create_metrics(self, criterion=None, ignore_index=None):
        # FIXME : Don't call super().create_metrics because its unfortunaly made for classification...
        metrics = {}
        metrics["scalars"] = {f"{self.name}_mse": utils.metrics.MSEWithLogits(output_transform=self.output_transform),
                              f"{self.name}_loss": utils.metrics.Loss(criterion, output_transform=self.output_transform)}
        metrics["summary"] = {}
        metrics["summary"][f"{self.name}_mse"] = metrics["scalars"][f"{self.name}_mse"]
        metrics["update"] = list(metrics["scalars"].values())
        metrics["matrices"] = {}
        self.metrics = metrics

class Mask(Output):
    """ Handles the Mask segmentation output type. Computes a confusion matrix for easier
        IoU computation.
    """
    def __init__(self, i, shape, classes, **kwargs):
        assert type(classes) == list, f"Mask outputs needs classnames, but got {classes}"
        super().__init__(i, shape, len(classes)+1, **kwargs)
        self.classes = classes

    def create_metrics(self, criterion=None, ignore_index=None):
        super().create_metrics(criterion, ignore_index)
        output_transform = self.output_transform
        cm = utils.metrics.ConfusionMatrix(self.n_outputs, output_transform=output_transform)
        cm_recall =  utils.metrics.MetricsLambda(lambda x:x/(x.sum(dim=1) + 1e-15), cm)
        cm_precision = utils.metrics.MetricsLambda(lambda x:x/(x.sum(dim=0) + 1e-15), cm)
        miou = utils.metrics.mIoU(cm, ignore_index=ignore_index)
        iou = utils.metrics.IoU(cm)
        self.metrics["update"].append(cm)
        self.metrics["scalars"][f"{self.name}_miou"] = miou
        self.metrics["summary"][f"{self.name}_miou"] = miou
        self.metrics["matrices"][f"{self.name}_recall"] = cm_recall
        self.metrics["matrices"][f"{self.name}_precision"] = cm_precision
        for ii, name in enumerate(["bg", *self.classes]):
            i = ii
            print(f"{name}, {i}")
            self.metrics["scalars"][f"{self.name}_iou_{name}"] = utils.metrics.MetricsLambda(lambda x, i=i: x[i], iou)
            self.metrics["scalars"][f"{self.name}_recall_{name}"] = utils.metrics.MetricsLambda(lambda x, i=i: x[i][i], cm_recall)
            self.metrics["scalars"][f"{self.name}_precision_{name}"] = utils.metrics.MetricsLambda(lambda x, i=i: x[i][i], cm_precision)

class Class(Output):
    """ Handles the classification output. Adds precision and recall for more precise
        per-class information
    """
    def __init__(self, idx, shape, classes, **kwargs):
        super().__init__(idx, shape, len(classes), **kwargs)
        self.classes = classes

    def create_metrics(self, criterion, ignore_index):
        super().create_metrics(criterion, ignore_index)
        recall = utils.metrics.Recall(output_transform=self.output_transform)
        precision = utils.metrics.Precision(output_transform=self.output_transform)
        self.metrics["update"] += [recall, precision]
        self.metrics["scalars"].update({
            f"{self.name}_precision":utils.metrics.MetricsLambda(lambda x: torch.mean(x), precision),
            f"{self.name}_recall":utils.metrics.MetricsLambda(lambda x: torch.mean(x), recall)
            })

        self.metrics["summary"].update({
            f"{self.name}_precision":utils.metrics.MetricsLambda(lambda x: torch.mean(x), precision),
            f"{self.name}_recall":utils.metrics.MetricsLambda(lambda x: torch.mean(x), recall)
            })
        for i, name in enumerate(self.classes):
            self.metrics["scalars"].update({
                f"{self.name}_precision_{name}":utils.metrics.MetricsLambda(lambda x, i=i: x[i], precision),
                f"{self.name}_recall_{name}":utils.metrics.MetricsLambda(lambda x, i=i: x[i], recall)
            })


class OutputList(list):
    """ Overwrites list for Outputs to more easily propagate update and computation calls
    """
    def __init__(self, criterion, batch_size=lambda x:x.shape[0]):
        super().__init__()
        self.loss = utils.metrics.Loss(criterion, batch_size=batch_size)

    def reset(self):
        self.loss.reset()
        for output in self:
            output.reset()

    def update(self, output):
        self.loss.update(output)
        for outp in self:
            transformed_output = outp.output_transform(output)
            outp.update(transformed_output)

    def compute(self):
        outputs = {"loss": self.loss.compute()}
        for output in self:
            outputs = {**outputs, **output.compute()}
        return outputs

    def summary(self):
        outputs = {"loss": self.loss.compute()}
        for output in self:
            outputs = {**outputs, **output.summary()}
        return outputs


def parse_outputs(outputs, **kwargs):
    """ Creates Output objects based on a list of outputs

    Arguments:
        outputs (list): list of strings each representing one output of the neural network,
                        it should contain type_suffix_width
        kwargs (dict): a dictionary of default values used for the outputs init

    Returns:
        list of Outputs that has the same length as outputs
    """

    supported_outputs = {
        "mask": Mask,
        "keypoints": Keypoints,
        "class": Class
    }

    if type(outputs) != list:
        outputs = [outputs]
    result = []
    for i, out_name in enumerate(outputs):
        out_split = out_name.split('_')
        n_split = len(out_split)
        if n_split > 3:
            raise ValueError(f"Unsupported output {out_name}: too many members")
        elif n_split == 3:
            out_type, type_suffix, width = out_split
            width = int(width)
        else: # < 3
            if n_split == 2:
                out_type, _uk = out_split
                try: # determine if type_suffix or width were given
                    width, type_suffix = int(_uk), None
                except:
                    type_suffix, width = _uk, None
            else: # == 1
                out_type = out_split[0]
                type_suffix, width = None, None

            if type_suffix is None:
                type_suffix = ""
            if width is None:
                width = kwargs[out_type]
        result.append(supported_outputs[out_type](
            i, (1,1), width, type_suffix=type_suffix
        ))
    return result

