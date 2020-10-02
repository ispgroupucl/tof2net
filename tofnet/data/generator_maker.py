from tofnet.data.generators import TrainLoader, Pipeline
from tofnet.data import datasets
from tofnet.data import samplers
from tofnet.data.datasets import DefaultDataset, ConcatDataset
from tofnet.data.preprocess import get_resize, normalize, get_device
from tofnet.data.augmentations import RandomScalingCropping, RandomFlipping, RandomRotation, RandomSize
from tofnet.utils.io import get_datadir
import os
from os import path
import numpy as np
from torch.utils.data import Subset
from tofnet.utils.config import get_class
import random
import torch
from tqdm.autonotebook import tqdm
from pathlib import Path
from copy import deepcopy


class NamedSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.np_indices = np.array(indices)
        self.shape = dataset.shape

    def get_filenames(self, indices):
        if type(indices) == int:
            indices = [indices]
        return self.dataset.get_filenames(self.np_indices[np.array(indices)])
    def __getattr__(self, name):
        return getattr(self.dataset, name)

    @staticmethod
    def collate_fn(batch):
        return DefaultDataset.collate_fn(batch)

def get_subset(dataset, n_items, seed=123456):
    rr = random.Random(seed)
    samples = sorted(rr.sample(range(len(dataset)), min(n_items, len(dataset))))
    return NamedSubset(dataset, samples)


def get_generators(cfg, act_time, n_classes, inputs, outputs, create_logs=True, nvis_train=0, nvis_val=0):
    """ Based on the cfg, creates the data-generators used to train/val/test the network.
    """
    data_dir = get_datadir()

    # Generate data-paths
    base_path  = path.join(data_dir, cfg["dataset"]["name"])
    tr_path = path.join(base_path, "train")
    val_path = path.join(base_path, "val")
    te_img_path = path.join(base_path, "test")
    te_lab_path = path.join(cfg["saves"]["path"], act_time)
    if create_logs is True:
        if not path.exists(te_lab_path):
            os.makedirs(te_lab_path)

    # Extract dims
    dims = cfg["dataset"]["dimensions"]
    test_dims = dims
    if 'test_dataset' in cfg:
        test_dims = cfg["test_dataset"].get("dimensions", dims)
    val_dims = test_dims
    if 'val_dataset' in cfg:
        val_dims = cfg["val_dataset"].get("dimensions", test_dims)

    if not cfg["dataset"].get("channels_first", False):
        dims = (dims["2"], dims["0"], dims["1"])
        cfg["dataset"]["dimensions"] = dims
        val_dims = (val_dims["2"], val_dims["0"], val_dims["1"])
        if "val_dataset" in cfg:
            cfg["val_dataset"]["dimensions"] = val_dims
        test_dims = (test_dims["2"], test_dims["0"], test_dims["1"])
        if "test_dataset" in cfg:
            cfg["test_dataset"]["dimensions"] = test_dims
        cfg["dataset"]["channels_first"] = True

    def add_default_dtypes(args):
        if "dtypes" not in args:
            args["dtypes"] = ["image", "mask"]
        return args

    if "part" in cfg["dataset"]: # multi-part dataset from different directories
        parts = deepcopy(cfg["dataset"]["part"])
        train_dss = []
        val_dss = []
        test_dss = []
        for part in parts:
            train_dss.append(get_class(add_default_dtypes({"root_dir":tr_path, **cfg["dataset"], **part}), datasets))
            # Order : val_dataset part overwrites train dataset
            val_dss.append(get_class(add_default_dtypes({"root_dir":val_path, **cfg["dataset"], **part, **cfg.get("val_dataset", {})}), datasets))
            test_dss.append(get_class(add_default_dtypes({"root_dir":te_img_path, **cfg["dataset"], **part, **cfg.get("test_dataset", {}), "only_labeled":False}), datasets))

        train_ds = ConcatDataset(train_dss)
        val_ds = ConcatDataset(val_dss)
        test_ds = ConcatDataset(test_dss)
        train_ds.collate_fn = val_ds.collate_fn = test_ds.collate_fn = datasets.DirectoryDataset.collate_fn
        sampler = get_class({"type": cfg["dataset"].get("sampler", "RandomSampler"), "data_source": train_ds}, samplers)
    else:
        # Make Dataset objects from paths
        train_ds = get_class(add_default_dtypes({"root_dir":tr_path, **cfg["dataset"]}), datasets)
        test_ds  = get_class(add_default_dtypes({"root_dir":te_img_path, **cfg["dataset"], **cfg.get("test_dataset", {}), "only_labeled":False}), datasets)
        val_ds   = get_class(add_default_dtypes({"root_dir":val_path, **cfg["dataset"], **cfg.get("test_dataset", {}), **cfg.get("val_dataset", {})}), datasets)
        sampler = get_class({"type": cfg["dataset"].get("sampler", "RandomSampler"), "data_source": train_ds}, samplers)

    # Pipelines for img augmentation + preprocessing
    pipe = []
    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"
    pipe.append(get_device(device))
    test_pipe = pipe.copy()
    val_pipe = pipe.copy()
    pipe.append(get_resize((dims[1], dims[2])))
    test_pipe.append(get_resize((test_dims[1], test_dims[2])))
    val_pipe.append(get_resize((val_dims[1], val_dims[2])))
    test_pipe  = Pipeline(test_pipe+[normalize])
    val_pipe   = Pipeline(val_pipe+[normalize]) # no DA

    aug_cfg = cfg["augmentation"]
    if "zoom_range" in aug_cfg:
        zr = aug_cfg["zoom_range"]
        pipe.append(RandomScalingCropping((max(1-zr, 0.8),1+zr),(dims[1], dims[2])))
    if "horizontal_flip" in aug_cfg:
        pipe.append(RandomFlipping(aug_cfg["horizontal_flip"]))
    if "rotation_range" in aug_cfg:
        pipe.append(RandomRotation(aug_cfg["rotation_range"]))
    if "RandomSize" in aug_cfg:
        pipe.append(RandomSize(**aug_cfg["RandomSize"]))
    train_pipe = Pipeline(pipe+[normalize]) # applies Data Augmentation

    # Make DataLoaders for datasets
    visualize = False if not create_logs else Path(te_lab_path) / "train_vis.pdf"
    batch_size = cfg["training"]["batch_size"] # Adapt batch_size in case val/test have larger inputs
    val_batch_size  = min(batch_size, max(1, int(batch_size * dims[2]/val_dims[2]  * dims[1]/val_dims[1])))
    test_batch_size = min(batch_size, max(1, int(batch_size * dims[2]/test_dims[2] * dims[1]/test_dims[1])))

    num_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 4))
    train_gen = TrainLoader(train_ds, inputs, outputs,
                            batch_size=batch_size,
                            pipeline=train_pipe,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=num_workers,
                            visualize=visualize)
    val_gen   = TrainLoader(val_ds, inputs, outputs,
                            batch_size=val_batch_size,
                            pipeline=val_pipe,
                            num_workers=num_workers,
                            shuffle=False)
    train_vis, val_vis = None, None
    if nvis_train > 0:
        train_vis_ds = get_subset(train_ds, nvis_train)
        train_vis    = TrainLoader(train_vis_ds, inputs, outputs,
                                batch_size=batch_size,
                                pipeline=val_pipe,
                                shuffle=False)
    if nvis_val > 0:
        val_vis_ds  = get_subset(val_ds, nvis_val)
        val_vis     = TrainLoader(val_vis_ds, inputs, outputs,
                                batch_size=val_batch_size,
                                pipeline=val_pipe,
                                shuffle=False)

    n_test_files = 0
    for f in os.listdir(te_img_path):
        if path.isfile(path.join(te_img_path, f)):
            n_test_files += 1

    test_gen = TrainLoader(test_ds, inputs, outputs,
                    batch_size=test_batch_size,
                    pipeline=test_pipe,
                    shuffle=False)

    tester = lambda model: ()
    return (train_gen, train_vis), (val_gen, val_vis), (test_gen, tester)
