import importlib
import numpy as np
import math
import torch
import torch.nn as nn

from multiprocessing import Pool
from tofnet.models.model_config import freeze_module
from tofnet.models.blocks import create_cba
from tofnet.models.model_config import ModelConfig
import time
import re
import pickle
from collections import OrderedDict
DEFAULT_PROTOCOL = 2


def get_model(cfg, classes=None):
    """Returns a model according to the specifics in cfg
        Also reloads weights if pretrained_weights is set in cfg

    Arguments:
        cfg (dict): the model's config
        classes (list): a list of the class names
    """
    # Getting the quantization if any
    complete_cfg = cfg
    model_cfg = ModelConfig(**cfg.get("quantization", {}))

    # Getting the right network + dimensions
    complete_cfg = cfg
    dims = cfg["dataset"]["dimensions"]

    if not cfg["dataset"].get("channels_first", False):
        dims = (dims["2"], dims["0"], dims["1"])
    cfg = cfg["network"].copy()
    network = cfg["type"]
    del cfg["type"]
    cfg.pop("n_params", None)
    cfg.pop("n_ops", None)

    cfg["input_size"] = dims
    cfg["classes"] = list(OrderedDict.fromkeys([classval for classval in classes if classval != ""]))
    pre_weights = cfg.pop("pretrained_weights", None)

    cfg["model_cfg"] = model_cfg

    module = importlib.import_module('.'+network, package="tofnet.models")
    network = getattr(module, network)
    model = network(**cfg)

    if pre_weights is not None:
        device = torch.device("cuda:0" if torch.cuda.device_count()>0 else "cpu")
        pretrained_model = torch.load(pre_weights, map_location=device)
        for pre in list(pretrained_model.keys()):
            if ".total_ops" in pre or ".total_params" in pre:
                del pretrained_model[pre]
        load_result = model.load_state_dict(pretrained_model, strict=False) # FIXME : don't use strict
        if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
            print(f"WARNING: missing or unexpected keys found")
            complete_cfg["network"]["missing_keys"] = load_result.missing_keys
            complete_cfg["network"]["unexpected_keys"] = load_result.unexpected_keys
    else:
        init_weights(model, complete_cfg)

    return model


def freeze_layers(model, cfg):
    """ Freezes the layers with a radix common to strings set the config in training.freeze

    Arguments:
        model (torch.nn.Module): a pytorch model
        cfg (dict): the config
    """
    freeze = cfg["training"].get("freeze", None)
    if freeze is None:
        return
    to_freeze = freeze["blocks"]
    if type(to_freeze) is not list:
        to_freeze = [to_freeze]
    for part in to_freeze:
        if not hasattr(model, part):
            raise AttributeError(f"The network doesn't posses a(n) '{part}' part")
        freeze_module(getattr(model, part))


@torch.no_grad()
def init_weights(model, cfg):
    """ Initializes the layers according to the config in training.weight_init

    Arguments:
        model (torch.nn.Module): a pytorch model
        cfg (dict): the config, weight_init types are either kaiming or xavier,
                    but normal distribution works best in both cases,
                    fan_mode should be set to fan_in for kaiming
    """
    params = cfg["training"].get("weight_init", {})

    # This part fixes pytorch buggy default implementation
    act = cfg.get("quantization", {}).get("act", "leaky_relu")
    if "leaky" in act or act == "":
        neg_slope = 0.01
        nonlin = "leaky_relu"
        sampling = "kaiming"
    elif "relu" in act:
        neg_slope = 0
        nonlin = "relu"
        sampling = "kaiming"
    elif "tanh" in act:
        neg_slope = 0
        nonlin = "tanh"
        sampling = "kaiming"
    else:
        print(f"Activation of type {act} is not supported yet")
    gain = nn.init.calculate_gain(nonlin, neg_slope)

    # Override default params
    gamma = params.get("gamma", 1.0)
    sampling = params.get("sampling", sampling)
    distribution = params.get("distribution", "normal")
    fan_mode = params.get("fan_mode", "fan_in")
    gain = params.get("gain", gain)

    assert sampling in ["kaiming", "xavier"]
    assert distribution in ["normal", "uniform"]
    assert fan_mode in ["fan_in", "fan_out", "fan_avg"]

    def custom_weights_init(m):
        # This custom part does things by the book and mirrors Keras'
        # implementation instead of the wonky pytorch one
        # Support for depthwise convolutions has also been added
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if isinstance(m, nn.Conv2d):
                ksize = m.kernel_size[0] * m.kernel_size[1]
                ksize = ksize / m.groups
                fan_out = m.out_channels * ksize
                fan_in = m.in_channels * ksize
            else:
                fan_out = m.out_features
                fan_in = m.in_features
            fan_avg = (fan_in + fan_out)/2

            if sampling == "xavier":
                std = gain/math.sqrt(fan_in+fan_out)
            elif sampling == "kaiming":
                fan = {
                    "fan_in": fan_in, "fan_out": fan_out, "fan_avg": fan_avg
                }[fan_mode]
                std = gain/math.sqrt(fan)

            if distribution == "normal":
                m.weight.data.normal_(0, std)
            else:
                limit = math.sqrt(3)*std
                m.weight.data.uniform_(-limit, limit)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(gamma)

        if hasattr(m, "bias") and hasattr(m.bias, "data"):
            m.bias.data.zero_()

    model.apply(custom_weights_init)