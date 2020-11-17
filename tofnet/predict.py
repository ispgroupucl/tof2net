import argparse
import open3d
from tofnet import train
from pathlib import Path, PosixPath
from tofnet.utils.config import Config
import pandas as pd
import zmq
from tofnet.utils.io import read_sample, write_sample
from tofnet.utils.torch_utils import collate_batch
from tofnet.data.generators import Pipeline
from tofnet.data.preprocess import get_device, get_resize, normalize, predict_preprocess
from tofnet.evaluation.pipeline import remove_channels, filter_formats
# from tofnet.pointcloud.configuration import PointCloudConfig
from tofnet.utils.pointcloud import PointCloudConfig
from tofnet.pointcloud.utils import transform_with_conf
from tofnet.models.model_maker import get_model
import numpy as np
import importlib
import copy
import struct
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
from pypcd import pypcd
import cv2
import sys
from multiprocessing import Pool

def create_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Evaluate network from directory")
    # parser.add_argument("directory", help="Path to model directory")
    parser.add_argument("--in_conf", help="path to a default kaspard.conf that will be read")
    parser.add_argument("--out_conf", help="path to write the computed kaspard.conf")
    parser.add_argument("--floor", help="Path to floor segmentation model directory, including config.toml and model.pt")
    parser.add_argument("--object", help="Path to object segmentation directory, including config.toml and model.pt")
    parser.add_argument("--pcd", help="path to a point cloud, don't set for daemon ZMQ mode")


def init(directory, pretrain=True):
    """ Initializes a model, a pipeline and returns the config object.

    Args:
        directory (Path): path to a directory containing a 'config.toml' file and
            weights for a model in the 'model.pt' file
        pretrain (bool): if False keeps weights random instead of loading weights

    Returns:
        tuple (torch.nn.Module,Pipeline,Conf):
            1. model network, and function you can call with
            correct input, which will give you the desired output.

            2. Pipeline used for preprocessing raw image

            3. configuration file read from 'config.toml'

    .. note:: In most cases, use predict and predict_batch with the returned tuple

    """

    cfg = Config(configfile=directory / "config.toml")

    # Set weight file
    if pretrain:
        weightfile = directory / "model.pt"
        cfg["network"]["pretrained_weights"] = weightfile

    # Get dimensions from test_dataset or (if not present) dataset
    dims = cfg["dataset"]["dimensions"]
    test_dims = cfg["test_dataset"].get("dimensions", dims)

    # Create pipeline for dataloader
    pipe = []
    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"
    pipe.append(predict_preprocess())
    pipe.append(get_device(device))
    pipe.append(get_resize((test_dims[1], test_dims[2])))
    pipe = Pipeline(pipe+[normalize])

    # Model creation
    model = get_model(cfg, classes=cfg["dataset"]["classes"])
    model.eval()
    model.to(device)
    cfg["test_dims"] = test_dims

    return model, pipe, cfg


def create_from_inputformat(sample, input_format):
    """ Load module using the input_format information.

    Args:
        sample (dict): dict-like object with at least a `pcd` key
        input_format ([str]): list of strings where the first part of each
            element contains the input type and the rest more information

    Returns:
        sample (dict): same dict object with elements added, according to input_format
    """
    for form in input_format:
        category = form.split("_", 1)[0]
        module = importlib.import_module(f"tofnet.annotations.{category}")
        inverse_names = getattr(module, "inverse_names")
        iform = inverse_names(form)
        style = None if len(iform.split("_",1))==1 else iform.split('_',1)[1]
        sample[form] = getattr(module, "generate_sample")(style, sample)
    return sample

def predict_batch_grad(model_triple, full_samples):
    model, pipe, cfg = model_triple
    input_format = cfg["network"]["input_format"]
    input_format = remove_channels(input_format)
    samples = []
    for sample in full_samples:
        dico = {}
        for dtype, data in sample.items():
            if dtype not in input_format:
                continue
            dico[dtype] = data
        samples.append(dico)

    piped_samples = []
    for sample in samples:
        piped_sample = {}
        for form in input_format:
            piped_sample[form] = pipe(sample[form], form.split("_",1)[0], sample)
        piped_samples.append(piped_sample)
    piped_samples = collate_batch(piped_samples)
    for _, v in piped_samples.items():
        v.requires_grad = True
    segmentation = model(piped_samples)[0]
    with torch.no_grad():
        decision = torch.zeros_like(segmentation).to(torch.long)
        decision[:,2,:,:] = (torch.argmax(segmentation)==2).to(torch.long)
    loss = torch.mean(-torch.log(
        1-torch.softmax(segmentation, dim=1)[:,2,:,:]
    ))
    loss.backward()
    grads = {}
    for k, v in piped_samples.items():
        grads[k] = v.grad.cpu()
    return segmentation.detach().cpu(), grads

@torch.no_grad()
def predict_batch(model_triple, confs, pcds, interpolate=True, full_samples=None, multi_thread=False, n_infers=1):
    """ Uses model (created by :meth:`init`) to predict on a batch of images.

    Args:
        model_triple (tuple): object containing model, pipeline and mod conf.
        confs (list[dict]): configuration for floor (if predicting floor,
                            leave blank ({}))
        pcds (list[dict]): list of pcds from read_sample
        interpolate (bool): activate interpolation to match pcd shape
        full_samples (list[dict]): complete samples from read_sample,
                                    or output from a previous predict_batch
        multi_thread (bool): use a process-pool instead of numba
        n_infers (int): number of inference for mcdropout

    Returns:
        tuple (:class:`torch.Tensor`,list[dict]):
            model:
            batch model output in one-hot encoding (B,C,H,W).

            sample:
            Computed samples necessary for model

    Example::

        >>> model_triple = init("path_to_model")
        >>> pcd = read_sample(pcd="path_to_pcd")
        >>> segmentation, out_samples = predict_batch(model_triple, [{}], [pcd])
        >>> segmentation.shape
        torch.Size([1, 3, 120, 160])
        >>> list(out_samples[0].keys())
        ['pcd', 'conf', 'image', 'depth_xyztcoZzL']

    """
    model, pipe, cfg = model_triple
    input_format = cfg["network"]["input_format"]
    input_format = remove_channels(input_format)
    if full_samples is None:
        read_fx = (lambda x: x) if type(pcds[0]) != PosixPath else read_sample
        pcd_confs = [
            read_fx({"pcd": pcd, "conf": conf}) for conf, pcd in zip(confs, pcds)
        ]
        if multi_thread:
            pool = Pool(6)
            samples = pool.starmap(create_from_inputformat, [
                (pcd_conf, input_format) for pcd_conf in pcd_confs
            ])
        else:
            samples = []
            for pcd_conf in pcd_confs:
                samples.append(create_from_inputformat(pcd_conf, input_format))
    else:
        samples = []
        for sample in full_samples:
            dico = {}
            for dtype, data in sample.items():
                if dtype not in input_format:
                    continue
                dico[dtype] = data
            samples.append(dico)

    piped_samples = []
    for sample in samples:
        piped_sample = {}
        for form in input_format:
            piped_sample[form] = pipe(sample[form], form.split("_",1)[0], sample)
        piped_samples.append(piped_sample)
    piped_samples = collate_batch(piped_samples)
    if n_infers == 1:
        segmentation = model(piped_samples)[0].detach().cpu()
    else:
        model.apply(lambda x: x.train() if type(x)==torch.nn.Dropout2d else x)
        segmentation = []
        for _ in range(n_infers):
            segmentation.append(model(piped_samples)[0].detach().cpu().unsqueeze_(0))
        segmentation = torch.cat(segmentation, dim=0)
        model.apply(lambda x: x.eval() if type(x)==torch.nn.Dropout2d else x)

    if interpolate:
        # FIXME support interpolate in other cases
        segmentation = F.interpolate(
            segmentation, size=list(reversed(pcds[0]["shape"])),
            mode="bilinear", align_corners=False
        )
    return segmentation, samples

@torch.no_grad()
def predict(model_triple, conf, pcd, transform=True):
    """ Predict for one image. See `predict_batch`."""
    segmentation, samples = predict_batch(model_triple, [conf], [pcd])
    if transform:
        segmentation = torch.argmax(segmentation[0], axis=0)
    # plt.imshow(sample["image"])
    # plt.imshow(segmentation, alpha=0.7)
    # plt.show()
    return segmentation, samples[0]

def predict_floor(model_triple, conf, pcd):
    """ Predict the configuration of the floor.

    Args:
        model_triple (tuple): contains all model information needed
        conf (dict): should be empty in most cases, for api compat
        pcd (np.array): sensor information

    Returns:
        conf: adds the camera section to the configuration dict
    """
    floor_segmentation, sample = predict(model_triple, conf, pcd)
    pcl_cfg = PointCloudConfig(copy.deepcopy(sample["pcd"]))
    centroid, normal, fitness = pcl_cfg.find_floor(floor_segmentation, threshold=0.1, min_samples=6)
    pcd_out, cam_config = pcl_cfg.rotate_gravity(normal, centroid)
    conf = {**conf, **cam_config}
    return conf

def predict_object(model_triple, conf, pcd):
    """Predict the configuration of the bed. See `predict_floor`."""
    object_segmentation, sample = predict(model_triple, conf, pcd)
    pcd = copy.deepcopy(sample["pcd"]) # ["points"]
    pcd["points"] = transform_with_conf(pcd["points"], conf, do_bed_transform=False)
    pcl_cfg = PointCloudConfig(pcd)
    object_segmentation = object_segmentation # .cpu().numpy()
    obj_conf = pcl_cfg.find_bed(object_segmentation, method="no_icp")
    conf = {**conf, **obj_conf}
    # FIXME
    conf["bed"]["orientation"] += 90
    conf["bed"] = [conf["bed"]]
    return conf


def main(args):
    """ DEPRECATED. Predict images sent through ZeroMQ or a pcd file.

    Listens to port 5555 for REP/REQ zmq messages of the form:

    Args:
        pcd (PointCloud): pcl-like point-cloud in binary form

    Returns:
        result (PointCloud): pcl-like point-cloud with annotation for every point
            in the pointcloud

    """
    # pylint: disable=no-member
    if args.floor is None and args.object is None:
        print("You should specify at least one model to run !")
        exit(1)
    if args.floor:
        floor_directory = Path(args.floor)
        floor_model = init(floor_directory)
    if args.object:
        object_directory = Path(args.object)
        object_model = init(object_directory)

    # test_dims = floor_cfg["test_dims"]

    if args.pcd:
        # Evaluate model directly with ONE pcd
        samplepaths = {"pcd": args.pcd}
        if args.in_conf:
            samplepaths["conf"] = args.in_conf
        sample = read_sample(samplepaths)
        if args.floor:
            config = predict_floor(floor_model, {}, sample["pcd"])
        if args.object:
            config = predict_object(object_model, config, sample["pcd"])
        if args.out_conf:
            write_sample({'conf':config}, {'conf':args.out_conf})
        else:
            print(config)

    else:
        raise NotImplementedError("Only one-pcd mode is currently available.")
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind("tcp://*:5555")


def annotate_pcd(pcd, annotation):
    """Convert annotation to pointcloud."""
    print(annotation.shape)
    print(pcd.pc_data["intensity"].shape)
    print(pcd.height, pcd.width)
    annotation = cv2.resize(annotation[0], (pcd.width, pcd.height))
    print(annotation.shape)
    # pcd.pc_data["intensity"] = np.flip(annotation, (0,1)).reshape(annotation.size)
    pc_data = np.empty(annotation.size, dtype=[('segmentation', 'u1')])
    pc_data["segmentation"] = np.flip(annotation, (0,1)).reshape(annotation.size)
    md = {'version': .7,
          'fields': ['segmentation'],
          'size': [1],
          'type': ['U'],
          'count': [1]}
    pcd = pypcd.add_fields(pcd, md, pc_data)
    return pcd
