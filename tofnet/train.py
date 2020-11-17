import argparse
import inspect
import json
import logging
import math
import os
import sys
import warnings
from collections import OrderedDict
from datetime import datetime
from functools import reduce
from math import ceil
from os import path
from pathlib import Path
from itertools import cycle

import gpustat
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as OPT
import torch.optim.lr_scheduler as LRS

from tofnet import utils
from tofnet.data.generator_maker import get_generators
from tofnet.models.model_maker import get_model, freeze_layers
from tofnet.training.callbacks import (CallbackList, KapImageCheckpoint,
                                       KapModelCheckpoint, ProgressCallback, CSVLogger)
from tofnet.training.losses import Criterion, MultiCriterion
import tofnet.training.losses as Losses
from tofnet.utils.config import get_class, get_config, Config
from tofnet.utils.print_utils import std_out_err_redirect_tqdm
from tofnet.utils.counter import profile as get_model_stats
from tofnet.utils.outputs import OutputList
from tofnet.utils.metrics import MetricList


def _vis_and_exit(image_gen):
    """Debug function to check the correct images are fed to the network
    """
    for image, target in image_gen:
        plt.imshow(image[0].cpu().permute(1,2,0)[:,:,0])
        plt.show()
        plt.imshow(target[0].cpu())
        plt.show()
    exit(-1)

class Train:
    def __init__(self, model: nn.Module, train_loader, val_loader, test_loader, cfg, metrics=None):
        self.device = torch.device("cuda:0" if torch.cuda.device_count()>0 else "cpu")
        self.model = model.float().to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg

        self.epochs = cfg["training"]["n_epochs"]
        self.batch_accumulation = cfg["training"].get("accumulation", 1)
        self.steps = math.ceil(len(train_loader) / self.batch_accumulation)

        self.metrics = metrics or MetricList()
        self.callbacks = CallbackList()

        # Import loss and class-weights
        cweights = cfg["dataset"].get("weights", {})
        class_weights = []
        for output in self.model.outputs:
            weight = cweights.get(output.name, None)
            weight = weight if weight is None else torch.Tensor(weight).to(self.device)
            class_weights.append(weight)
        print(class_weights)
        _loss = cfg["training"]["loss"]
        if type(_loss) is not list:
            _loss = [_loss]
        self.criterion = []
        for ll, llcweights in zip(_loss, cycle(class_weights)):
            if type(ll) != str: # Loss can be a dictionary
                self.criterion.append(Criterion(get_class(ll, Losses, weight=llcweights, device=self.device, ignore_index=255)))
            else:
                self.criterion.append(Criterion(get_class({"type": ll}, Losses, weight=llcweights, ignore_index=255)))
        _loss_weights = cfg["training"].get("weights", None)
        if _loss_weights is not None:
            if type(_loss_weights) is not list:
                _loss_weights = [_loss_weights]
            _loss_weigths = torch.Tensor(_loss_weights).to(self.device)
        self.criterion = MultiCriterion(self.criterion, _loss_weights)

        # Import optimizer
        _optimizer = cfg["training"]["optimizer"].copy()
        _optimizer["params"] = list(filter(
            lambda npa: npa[1].requires_grad, self.model.named_parameters()
        )) # filter out frozen layers

        if "layca" in _optimizer["type"].lower():
            _optimizer["params"] = [
                {"params": list(
                    p for n, p in _optimizer["params"] if n.endswith(".bias")
                ), "layca": False # layca can't train biasses
                }, {"params": list(
                    p for n, p in _optimizer["params"] if n.endswith(".weight")
                ), "layca": True
                }
            ]
        else:
            _optimizer["params"] = [pa for _, pa in _optimizer["params"]]
        self.optimizer = get_class(_optimizer, OPT)

        # Import lr scheduler
        _lr_sched = cfg["training"].get("scheduler", None)
        if _lr_sched is not None:
            _lr_sched = _lr_sched.copy()
            _lr_sched["optimizer"] = self.optimizer
            self.lr_sched = get_class(_lr_sched, LRS)

    def register_callbacks(self, callbacks):
        for callback in callbacks:
            self.callbacks.append(callback)

    def set_metrics(self, metrics):
        self.metrics = metrics

    def fit(self):
        self.callbacks.set_model(self.model)
        self.callbacks.set_learner(self)
        self.callbacks.set_params({
            'epochs':self.epochs,
            'steps':self.steps,
            'verbose':True, # TODO : set as param
            'do_validation':True, # TODO : set as param
        })
        self.callbacks.on_train_begin()
        fit_pbar = range(self.epochs) # , desc="Epoch")
        for epoch in fit_pbar:
            self.callbacks.on_epoch_begin(epoch)
            train_metrics = self.train(epoch)
            val_metrics = self.validate(self.val_loader)
            for name, value in val_metrics.items():
                train_metrics["val_"+name] = value
            self.callbacks.on_epoch_end(epoch, logs=train_metrics)

        self.callbacks.on_train_end()

    def test(self):
        self.callbacks.set_model(self.model)
        self.callbacks.set_learner(self)
        self.callbacks.set_params({
            'epochs':self.epochs,
            'steps':self.steps,
            'verbose':True, # TODO : set as param
            'do_validation':True, # TODO : set as param
        })

        self.callbacks.on_test_begin()
        test_metrics = self.validate(self.test_loader)
        self.callbacks.on_test_end(logs=test_metrics)
        return test_metrics


    def save_checkpoint(self, path):
        if self.cfg is not None:
            torch.save(self.model.state_dict(), Path(self.cfg["saves"]["path"]) / 'model.pt')

    def train(self, epoch):
        self.model.train()
        train_pbar = self.train_loader # , desc="Train")

        self.metrics.reset()
        self.optimizer.zero_grad()
        max_iter = len(self.train_loader)-1
        for i, (images, target) in enumerate(train_pbar):
            if i % self.batch_accumulation == 0:
                self.callbacks.on_train_batch_begin(i, logs={'size':images[list(images.keys())[0]].shape[0]})

            output = self.model(images)
            self.metrics.update((output, target))
            loss = self.criterion(output, target)
            loss.backward()
            if i % self.batch_accumulation == self.batch_accumulation-1 or i == max_iter:
                self.optimizer.step()
                self.optimizer.zero_grad()
                computed_metrics = self.metrics.compute()
                self.callbacks.on_train_batch_end(i, logs=computed_metrics)

        if hasattr(self, 'lr_sched'):
            self.lr_sched.step(epoch)

        return self.metrics.compute()


    def validate(self, val_loader):
        self.model.eval() # set model in evaluation mode
        self.metrics.reset()
        with torch.no_grad():
            losses = []
            for i, (images, target) in enumerate(val_loader):
                self.callbacks.on_test_batch_begin(i, logs={'size':images[list(images.keys())[0]].shape[0]})
                output = self.model(images)
                loss = self.criterion(output, target)
                self.metrics.update((output, target))
                losses.append(loss)
                computed_metrics = self.metrics.compute()
                self.callbacks.on_test_batch_end(i, computed_metrics)
        return self.metrics.compute()

    def predict(self, test_loader):
        self.model.eval()
        self.callbacks.on_predict_begin()
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                self.callbacks.on_predict_batch_begin(i, logs={'size':images.shape[0]})
                output = self.model(images)
                self.callbacks.on_predict_batch_end(i)

        self.callbacks.on_predict_end()
        return self.metrics.compute()

def train(config_file=None, config=None, eval_only=False, wfile=None, act_time=None, create_logs=True, args=None):
    results = {}
    if act_time==None:
        act_time = "{:%Y-%m-%dT%H%M%S}".format(datetime.now())
    try:
        cfg = Config(configfile=config_file, config=config, act_time=act_time) # get_config(config_file)
        act_time = cfg.name
        if create_logs:
            os.makedirs(path.join(cfg["saves"]["path"], act_time), exist_ok=True)
        if args.tensorboard:
            import subprocess
            tb = subprocess.Popen(["tensorboard", "--logdir="+path.join(cfg["saves"]["path"], act_time)], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        if create_logs:
            cfg.dump(path.join(cfg["saves"]["path"], act_time, "base_config.toml"))
        for config in cfg:
            act_time = config["saves"].get("act_time", act_time)
            eval_only = config.get("eval_only", eval_only)
            result = train_network(config, eval_only, wfile, act_time, create_logs, args=args)
            results[cfg.name] = result
    except KeyboardInterrupt:
        if args.tensorboard:
            tb.kill()
        exit(0)
    return results

def train_network(cfg, eval_only=False, wfile=None, act_time=None, create_logs=True, args=None):
    # from utils.domadapt import transfer_bns

    from numpy.random import seed
    seed(1)
    torch.manual_seed(1)

    if act_time is None and create_logs is True:
        raise ValueError("act_time should never be none if create_logs is true")

    cfg["act_time"] = act_time
    # Initialize data generators
    if wfile is not None:
        cfg["network"]["pretrained_weights"] = wfile

    if "type" not in cfg["training"]:
        cfg["training"]["type"] = "default"

    n_classes = len(cfg["dataset"]["classes"])
    bg_class = cfg["dataset"].get("add_bg", True)
    if n_classes != 1 or bg_class:
        n_classes += 1

    # Initialize model
    print(cfg["dataset"]["classes"])
    model = get_model(cfg, classes=cfg["dataset"]["classes"])
    try:
        n_ops, n_params, model_info =  get_model_stats(
            model, cfg=cfg, logpath=(
                Path(cfg["saves"]["path"]) / act_time
                if create_logs else
                None
            )
        )
        cfg["network"]["n_params"] = n_params
        cfg["network"]["n_ops"] = n_ops
        print(model_info)
    except:
        print("Error while computing Model statistics\n Execution will continue")
    out_names = model.outputs

    # Data generators
    (train_gen, train_vis_gen), (val_gen, val_vis_gen), (test_gen, tester) = \
            get_generators(cfg, act_time, n_classes, inputs=model.input_format, outputs=out_names,
                            create_logs=create_logs, nvis_train=args.vis_train,
                            nvis_val=args.vis_val)

    if create_logs:
        cfg.dump(path.join(cfg["saves"]["path"], act_time, "config.toml"))

    # Freezes the layers according to cfg
    freeze_layers(model, cfg)

    # Create Training object
    training = Train(model, train_gen, val_gen, test_gen, cfg=cfg)

    # Create Metrics
    metrics = OutputList(training.criterion, batch_size=lambda x:x[0].shape[0])
    for i, output in enumerate(model.outputs):
        output.create_metrics(training.criterion[i], cfg["dataset"].get("ignore_index", None))
        metrics.append(output)

    training.set_metrics(metrics)

    # Train model
    callbacks = []
    if create_logs:
        log_dir = Path(cfg["saves"]["path"])
        os.makedirs(log_dir, exist_ok=True)

        save_monitor = cfg["saves"].pop("monitor", None)
        save_monitor = save_monitor if save_monitor is not None else "val_loss"
        save_mode = cfg["saves"].pop("mode", "auto")

        model_path = log_dir / act_time / 'model.pt'
        model_checkpoint = KapModelCheckpoint(str(model_path),
                                        monitor=save_monitor,
                                        save_weights_only=cfg["saves"].get("save_weights_only", True),
                                        save_best_only=cfg["saves"]["save_best_only"],period=1,
                                        mode=save_mode)
        callbacks += [model_checkpoint]

        image_checkpoint = KapImageCheckpoint(log_dir / act_time,
                                train_vis_gen, val_vis_gen, test_gen, n_classes, period=1)
        callbacks += [image_checkpoint]

        model_logger = CSVLogger(Path(cfg["saves"]["path"])/act_time/"metrics.csv")
        callbacks += [model_logger]
    callbacks += [ProgressCallback(epochs=cfg["training"]["n_epochs"], train_steps=training.steps, val_steps=len(val_gen))]
    training.register_callbacks(callbacks)

    if not eval_only:
        # Create Callbacks, TODO : configuration of callbacks and put in callbacks.py

        # Train on dataset
        if args.return_fit:
            return training
        try:
            training.fit()
        except KeyboardInterrupt:
            pass

        if create_logs:
            try:
                model.load_state_dict(model_checkpoint.best_weights)
            except :
                print("Encountered an error when reloading best weights!!")

    # Save results
    # model = transfer_bns(model, test_gen, batch_size=cfg["training"]["batch_size"])
    return training.test()
    # tester(model, tensorboard=model_tensorboard)

def eval_network(config_file, wfile=None, act_time=None, create_logs=True, args=None):
    train(config_file=config_file, wfile=wfile, eval_only=True, act_time=act_time, create_logs=create_logs, args=args)

def create_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Train network using configuration file")
    parser.add_argument("config_file", help="Path to configuration file")
    parser.add_argument("--logdir", help="Name of output directory inside logs")
    parser.add_argument("--no-logs", action="store_true", help="Remove creation of logs directory. Use at your own risk ! (only for tests)")
    parser.add_argument("--vis-train", help="create train_images directory with the result of some images", default=0, type=int)
    parser.add_argument("--vis-val", help="create val_images directory with the result of some images", default=10, type=int)
    parser.add_argument("--return-fit", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")

    return parser

orig_stdout=None

def main(args):
    global orig_stdout
    # sanity check for stupid researchers
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpus = gpustat.new_query().jsonify()['gpus']
        if len(gpus) > 2:
            lowest_mem, lowest_index = (1000000, "")
            for gpu in gpus:
                if gpu['memory.used'] < lowest_mem:
                    lowest_mem = gpu['memory.used']
                    lowest_index = gpu['index']
            if lowest_mem > 2000:
                print("No GPU is available for now, try again later or leave a message after the tone *beep*")
                print("If you want to run on CPU, set CUDA_VISIBLE_DEVICES to the right value")
                exit(-2)
            lowest_index = str(lowest_index)
            os.environ["CUDA_VISIBLE_DEVICES"] = lowest_index
    with std_out_err_redirect_tqdm() as orig_stdout:
        if args.sub == "eval":
            eval_network(args.config_file, args.pretrained_model, act_time=args.logdir, create_logs=not args.no_logs, args=args)
        else:
            return train(config_file=args.config_file, create_logs=not args.no_logs, act_time=args.logdir, args=args)


if __name__ == "__main__":

    args = create_parser().parse_args()
    main(args)
