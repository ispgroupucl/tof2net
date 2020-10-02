import inspect
import torch
import json
import qtoml as toml
import collections
import types
import numpy as np
from collections import OrderedDict
from itertools import product
from pathlib import Path
from copy import deepcopy
from tofnet.training import strategies
from subprocess import check_output

default_strategies = [
    {"type": "Eval"},
    {"type": "Git"},
]

class BaseConfig:
    """ Presents a config file in an easier to manipulate format with support for either
        .json and .toml formats.
    """
    def __init__(self,*,configfile=None, config=None, act_time=None):
        if configfile is not None:
            configfile = Path(configfile)
            with open(configfile, 'r') as config:
                if configfile.suffix == ".toml":
                    self.config = toml.loads(config.read()) #  _dict=OrderedDict)
                    self.type = "toml"
                else:
                    self.config = json.load(config, object_pairs_hook=OrderedDict)
                    self.type = "json"
        elif config is not None:
            self.config = config
        else:
            raise ValueError("you should pass either a configfile or a config dict")

        self.act_time = self.name = act_time or ""
        if "type" in self.config:
            self.name += "_"+self.config["type"]

    def get_dict(self):
        return self.config

    def flatten(self, sep='.'):
        def _flatten(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if hasattr(v, "items"):
                    items.extend(_flatten(v, new_key).items())
                elif isinstance(v, list) and hasattr(v[0], "items"):
                    for i,vv in enumerate(v):
                        item_type = vv["type"]
                        items.extend(_flatten(vv, f"{new_key}{sep}{item_type}").items())
                else:
                    items.append((new_key, v))
            return dict(items)
        return _flatten(self.config)

    def inflate(self, sep='.'):
        def _inflate(d):
            items = dict()
            for k, v in d.items():
                keys = k.split(sep)
                sub_items = items
                for ki in keys[:-1]:
                    try:
                        sub_items = sub_items[ki]
                    except KeyError:
                        sub_items[ki] = dict()
                        sub_items = sub_items[ki]

                sub_items[keys[-1]] = v
            return items
        return _inflate(self.config)

    def __getitem__(self, key):
        return self.config[key]

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config


class Config(BaseConfig):
    """ Extends the BaseConfig object to add support for strategies. Some code can thus
        be executed before the true start of the training, enabling dataset creation or
        config modification to add support for grid-searches.
    """
    def __init__(self,*, configfile=None, config=None, act_time=None):
        super().__init__(configfile=configfile, config=config, act_time=act_time)
        self.strategies = []
        self.iterations = []
        for strategy in default_strategies:
            self.config = get_class(strategy, strategies, act_time=self.name)(self.config, 0)

        for strategy in self.config.get("strategy", []):
            strategy_class = get_class(strategy, strategies, act_time=self.name, config=self.config)
            self.strategies.append(strategy_class)
            self.iterations.append(range(len(strategy_class)))

    def dump(self, filename):
        config = deepcopy(self.config)
        if config.get("git", {}).get("diff", None) is not None:
            diff_filename = Path(filename).parent / "diff.patch"
            with open(diff_filename, 'w') as diff_file:
                diff_file.write(config["git"]["diff"])
            config["git"]["diff"] = str(diff_filename.resolve())
        with open(filename, 'w') as config_file:
            config_file.write(toml.dumps(config))

    def modify_config(self, config):
        for key, value in config.items():
            if isinstance(value, collections.Mapping) and "search" in value:
                if type(value["search"]) is list or value["search"] == "list":
                    search_list = value.get("list") or value["search"]
                    config[key] = np.random.choice(search_list)
                elif value["search"] == "distribution":
                    params = deepcopy(value)
                    del params["search"]
                    config[key] = get_class(params, np.random)
                else:
                    raise ValueError(value["search"]+" is not a possible value for hyperparams")
            elif isinstance(value, collections.Mapping):
                self.modify_config(value)

    def __iter__(self):
        for i in product(*self.iterations):

            config = deepcopy(self.config)
            if "strategy" in config:
                del config["strategy"]
            for s, strategy in enumerate(self.strategies):
                config = strategy(config, i[s])

            yield Config(config=config, act_time=self.name)
            for strategy in self.strategies:
                strategy.clean()


def get_config(infile):
    infile = Path(infile)
    with open(infile, 'r') as fp:
        if infile.suffix == ".toml":
            return toml.loads(fp.read())
        else:
            return json.loads(fp.read(), object_pairs_hook=OrderedDict)

def get_class(params, package, device=None, **optional):
    """ Extracts an object corresponding to params["type"] in package
        and initializes it with params (minus 'type')

    Arguments:
        params (dict): a dictionary in which the 'type' is present,
                    params['type'] determines which function will be
                    searched in package
        package (python module): the python module in which to search for the desired
                            function
        device (torch.device): if set, will transfer list-types to set device
        optional (dict): optional parameters that will only be added
                        if the function signature supports it
    """
    params = params.copy()
    fx_type = params["type"]
    del params["type"]
    my_obj = getattr(package, fx_type)
    for key, arg in optional.items():
        if key in inspect.signature(my_obj).parameters:
            params[key] = arg
    if device is not None:
        for key, item in params.items():
            if type(item) == str:
                continue
            if type(item) != list:
                item = [item]
            params[key] = torch.Tensor(item).to(device)
    return my_obj(**params)
