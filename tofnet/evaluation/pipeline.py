import sys
import os
import numpy as np
import open3d
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import skimage.io as io
import ipywidgets as widgets
from collections import defaultdict
from skimage import measure
import seaborn as sns
from tqdm import tqdm
import toml
from tofnet.utils import notebook_utils
from tofnet.utils.notebook_utils import Logpath, Roomdirs, DataSelection
from tofnet import predict
from tofnet.data import datasets
from tofnet.utils import pointcloud
import tofnet
from tofnet.utils.pointcloud import PointCloudConfig
from tofnet.pointcloud.metrics import floor_similarity, bed_similarity, full_similarity
from tofnet.pointcloud.utils import extract_template, transform_with_conf, fast_twconf, rotate
from tofnet.pointcloud.visualize import visualize_pointcloud
from tofnet.annotations.depth import inverse_names, generate_sample as depth_generate, compute_normals
from tofnet.data.datasets import DirectoryDataset
import torch
import torch.nn.functional as F
import copy
from collections import defaultdict as ddict
import pickle
import argparse
import re
from multiprocessing import Pool

def remove_channels(formats):
    return [
        x if len(x.split("_"))<=2 else x.rsplit("_", 1)[0]
        for x in formats
    ]

def filter_formats(sampledict, formats, styles=(), cheats=()):
    """
        Modifies @formats for compatibility reasons (erasing tmp seeds)
        and throws an error whenever a @cheat input is detected
    """
    for form in formats: # Compatibility with earlier tmp names
        for style in styles:
            if np.sum([style in cheat for cheat in cheats]) > 0:
                raise ValueError("You shouldn't be cheating at this stage")
            if style in form:
                suffix = form.split(style)[-1]
                if "_" not in suffix and len(suffix) == 6:
                    # print(f"changed {form} to {form[:-6]}")
                    sampledict[form] = sampledict[form[:-6]]

def get_templates(floor_roomdir, bed_roomdir, dataset):
    try:
        with open(f"cache/templates/{floor_roomdir.stem}.pickle", "rb") as template_fp:
            dir_templates = pickle.load(template_fp)
    except:
        template_samples = dataset.image_names(exclude=floor_roomdir.stem) # iter(exclude=floor_roomdir.stem)
        dir_templates = {}
        for sample_name in tqdm(template_samples):
            room = sample_name.rsplit("_", 1)[0]
            needed_samples = {"template", "conf"}
            sample_paths = dataset[sample_name]
            sample_paths = {key:sample_paths[key] for key in needed_samples}
            sample = datasets.read_sample(sample_paths)
            template = sample["template"] # extract_template(sample["pcd"]["points"], sample["conf"])
            if dir_templates.setdefault(room, (np.zeros((0,3)), None))[0].shape[0]<template.shape[0]:
                dir_templates[room] = (template, sample["conf"]["bed"])
        os.makedirs("cache/templates", exist_ok=True)
        with open(f"cache/templates/{floor_roomdir.stem}.pickle", "wb") as template_fp:
            pickle.dump(dir_templates, template_fp, pickle.HIGHEST_PROTOCOL)
    return dir_templates


def wrapped_findbed(args):
    pcl_cfg, bed_segmentation, templates, method, stem = args
    conf = pcl_cfg.find_bed(
        bed_segmentation, templates, method=method, home=stem
    ) or {}
    return conf

@torch.no_grad()
def avgIoU(trues, preds, idx):
    ious = []
    for i in range(trues.shape[0]):
        pred, true = preds[i].view(-1), trues[i].view(-1)
        pred, true = (pred==idx).to(torch.int), (true==idx).to(torch.int)
        intersection = torch.sum(pred*true).to(torch.float).numpy()
        union = torch.sum(torch.clamp(pred+true, 0, 1)).to(torch.float).numpy()
        ious.append(float(intersection/union))
    return ious

FOLD_REGEX = re.compile(r"Fold\d+--(.*)-room_(.*)_\d+")
def pipeline(floor_roomdir, batched_samplepaths, floor_triple,
             bed_triple, data_metrics, templates, args):
    samples, sample_cfgs, saved_outputs = [], [], []
    for samplepaths in batched_samplepaths:
        samples.append(datasets.read_sample(
            {key:samplepaths[key] for key in ["image", "pcd", "conf", "mask_normal"]}
        ))
        if args.no_model:
            im = cv2.imread(
                str(floor_roomdir / "images" / "test" / f'out_mask_normal3_{samplepaths["image"].stem}.png'),
                cv2.IMREAD_GRAYSCALE
            )
            saved_outputs.append(torch.tensor(im).unsqueeze_(0))

        # Wrapper around pcd for icp methods
        sample_cfgs.append(PointCloudConfig(copy.deepcopy(samples[-1]["pcd"])))
        # print(samplepaths["image"])

    # --------- Find Floor ---------
    if args.gt_floor:
        for sample, pcl_cfg in zip(samples, sample_cfgs):
            cam_config = {"camera":sample["conf"]["camera"]}
            fitness = 1
            pcl_cfg.pointcloud["points"] = fast_twconf(
                pcl_cfg.pointcloud["points"], cam_config
            )

    else:
        if args.ransac:
            floor_segmentation = torch.ones(len(samples), *sample_cfgs[0].ishape)
        else:
            _, _, floor_cfg = floor_triple
            # Find floor
            ## Change data
            floor_input_format =  remove_channels(floor_cfg["network"]["input_format"])
            floor_sample_names = set(floor_input_format) - {"image", "pcd", "conf", "mask_normal"}
            cat_samples = []
            for samplepaths, sample in zip(batched_samplepaths, samples):
                filter_formats(samplepaths, floor_input_format, styles=["xyz"], cheats=["height"])
                floor_samplepaths = {key:samplepaths[key] for key in floor_sample_names}
                cat_samples.append({**datasets.read_sample(floor_samplepaths), **sample})

            ## Segment pcd & resize to image shape
            if args.no_model:
                floor_segmentation = torch.cat(saved_outputs, dim=0).unsqueeze_(0).to(torch.float)
                floor_segmentation = F.interpolate(
                    floor_segmentation, size=sample_cfgs[0].ishape,
                    mode="nearest"
                )[0].to(torch.long)
            else:
                floor_segmentation, _ = predict.predict_batch(
                    floor_triple, None, None, interpolate=False, full_samples=cat_samples
                )
                floor_segmentation = F.interpolate(
                    floor_segmentation, size=sample_cfgs[0].ishape,
                    mode="bilinear", align_corners=True
                )
                floor_segmentation = torch.argmax(floor_segmentation, axis=1)

            if args.uncertain:
                floor_segmentations, _ = predict.predict_batch(
                    floor_triple, None, None, interpolate=False, full_samples=cat_samples, n_infers=50
                )
                floor_segmentation_var = torch.abs(torch.var(floor_segmentations, dim=0))
                floor_segmentation_var = F.interpolate(
                    floor_segmentation_var, size=sample_cfgs[0].ishape,
                    mode="bilinear", align_corners=True
                )[:,1]
                np_75 = np.percentile(floor_segmentation_var, args.percentile)
                is_floor = floor_segmentation==1
                floor_segmentation[is_floor & (floor_segmentation_var > np_75)] = 0


        ## Compute & apply rotation to set floor in z=0
        cam_configs, gt_confs = [], []
        for i, (pcl_cfg, sample) in enumerate(zip(sample_cfgs, samples)): # TODO multithread
            try:
                centroid, normal, fitness = pcl_cfg.find_floor(floor_segmentation[i], threshold=0.1, min_samples=6)
            except:
                centroid = np.array([0., -0.5, 3.0])
                normal = np.array([0., np.sqrt(2)/2, -np.sqrt(2)/2])
                fitness = 0
            _, cam_config_ = pcl_cfg.rotate_gravity(normal, centroid)
            cam_configs.append(cam_config_)

            # Compute floor difference metrics
            cam_config_["camera"]["cam_fitness"] = fitness
            cossim = floor_similarity(sample["conf"], cam_config_)
            sample["conf"]["bed"] = sample["conf"]["bed"][0] # account for list
            gt_confs.append(sample["conf"])
            sample["conf"] = cam_config_

            data_metrics["angle"][floor_roomdir.stem].append(cossim)
            data_metrics["gt_cfg"][floor_roomdir.stem].append(gt_confs[-1])



    # --------- Find bed ---------
    # Rotate pcd based on camera calibration
    gt_segmentation = torch.cat([
        torch.tensor(sample["mask_normal"]).unsqueeze(0) for sample in samples
    ])
    if args.gt_bed:
        bed_segmentation = torch.cat([
            torch.tensor(sample["mask_normal"]).unsqueeze(0) for sample in samples
        ])
    elif args.single_pass:
        bed_segmentation = floor_segmentation
        if args.uncertain:
            bed_segmentation_var = torch.abs(torch.var(floor_segmentations, dim=0))
            bed_segmentation_var = F.interpolate(
                bed_segmentation_var, size=sample_cfgs[0].ishape, mode="bilinear", align_corners=True
            )[:,2]
            np_75 = np.percentile(bed_segmentation_var, args.percentile)
            is_bed = bed_segmentation==2
            bed_segmentation[is_bed & (bed_segmentation_var > np_75)] = 0
    else:
        _, _, bed_cfg = bed_triple
        bed_input_format = remove_channels(bed_cfg["network"]["input_format"])
        for sample in samples:
            for iformat in bed_input_format:
                style = inverse_names(iformat)
                if style is not None:
                    ddd = depth_generate(style=style, sample=sample)
                    sample[iformat] = ddd.copy() # TODO: why is copy necessary to avoid negative strides error?

        bed_segmentation, _ = predict.predict_batch(bed_triple, None, None, interpolate=False, full_samples=samples)
        bed_segmentation = F.interpolate(
            bed_segmentation, size=sample_cfgs[0].ishape,
            mode="bilinear", align_corners=True
        )
        bed_segmentation = torch.argmax(bed_segmentation, axis=1)

        if args.uncertain:
            bed_segmentations, _ = predict.predict_batch(
                bed_triple, None, None, interpolate=False, full_samples=samples, n_infers=50
            )
            bed_segmentation_var = torch.abs(torch.var(bed_segmentations, dim=0))
            bed_segmentation_var = F.interpolate(
                bed_segmentation_var, size=sample_cfgs[0].ishape, mode="bilinear", align_corners=True
            )[:,2]
            np_75 = np.percentile(bed_segmentation_var, args.percentile)
            is_bed = bed_segmentation==2
            bed_segmentation[is_bed & (bed_segmentation_var > np_75)] = 0

    segious = avgIoU(gt_segmentation, bed_segmentation, idx=2)
    use_templates = args.use_templates
    method = args.method or "icp"

    if args.gt_bed or args.skip_loc:
        bed_confs = [{"bed": gt_conf["bed"]} for gt_conf in gt_confs]
    else:
        templates = templates if use_templates else None
        with Pool(10) as pool:
            bed_confs = pool.map(
                wrapped_findbed,
                [(pcl_cfg, bed_segmentation[i], templates, method, floor_roomdir.stem)
                 for i, pcl_cfg in enumerate(sample_cfgs)]
            )

    for bed_conf, samplepaths, sample, gt_conf, segiou in zip(bed_confs, batched_samplepaths, samples, gt_confs, segious):
        if len(bed_conf) > 0:
            pred_conf = {**bed_conf, **sample["conf"]}#**cam_config}
            # -- THIS IS A HACK, FIXME
            if use_templates and not args.gt_bed:
                raise ValueError("Unsupported for now")
                # if "bed" in pred_conf and "orientation" in pred_conf["bed"]:
                #     pred_conf["bed"]["orientation"] -= 90 # plus or minus??? from tests, I don't know
                #     ious, div_err, _, _, _ = full_similarity(sample["pcd"], sample_conf, pred_conf, sample=sample)
                #     cossim_bed, err_center, err_length, err_width = div_err
                # else:
                #     return
            # -- END HACK
            else:
                ious, div_err, _, _, _ = full_similarity(sample["pcd"], gt_conf, pred_conf, sample=sample)
                cossim_bed, err_center, err_length, err_width = div_err
        else:
            ious, cossim_bed, err_center, err_length, err_width, pred_conf = (0, 0),0,1,1,1, {**sample["conf"]}

        if args.save_toml:
            tomlpath = samplepaths["image"].parents[1] / "toml" / (samplepaths["image"].stem+".toml")
            tomlpath.parent.mkdir(parents=True, exist_ok=True)
            tomlconf = copy.deepcopy(pred_conf)
            for skey, sval in tomlconf.items():
                for key, val in sval.items():
                    tomlconf[skey][key] = float(val)
            tomlconf["bed"] = [tomlconf["bed"]]
            with tomlpath.open("w") as tomlfile:
                toml.dump(tomlconf, tomlfile)

        room_name = samplepaths["image"].stem
        rmatch = FOLD_REGEX.match(room_name)
        if rmatch is not None:
            room_name = f"{rmatch.group(1)}-r{rmatch.group(2)}"
        else: # Can be removed if no backwards compatibility is desired
            room_name = room_name.split("room_")[-1].split("_")[0]

        data_metrics["iou"][floor_roomdir.stem].append(ious[1])
        data_metrics["segiou"][floor_roomdir.stem].append(segiou)
        data_metrics["angle_bed"][floor_roomdir.stem].append(cossim_bed)
        data_metrics["room"][floor_roomdir.stem].append(room_name)
        data_metrics["pred_cfg"][floor_roomdir.stem].append(pred_conf)
        data_metrics["pcd"][floor_roomdir.stem].append(samplepaths["pcd"])
    return

def to_pandas(data_metrics):
    pdata = defaultdict(list)

    for k, rooms in data_metrics["room"].items():
        kk = [k]*len(rooms)
        pdata["house"] += kk
        pdata["id"] += [x+"_"+y for x,y in zip(kk, rooms)]
        for _metric in {*data_metrics.keys()}-{"room", "pred_cfg", "gt_cfg", "pcd"}:
            pdata[_metric] += data_metrics[_metric][k]
            _data = np.abs(data_metrics[_metric][k])
            if "angle" in _metric:
                over90 = _data>90
                _data[over90] = -_data[over90]+180

            pdata["abs_"+_metric] += list(_data)

    df = pd.DataFrame.from_dict(pdata)
    df = df.sort_values(['id'])

    return df

def make_plots(data_metrics, df):
    for _metric in {*data_metrics.keys()}-{"room", "pred_cfg", "gt_cfg", "pcd"}:
        plt.figure("abs_"+_metric,[10,15])
        sns.boxplot(x="abs_"+_metric, y="id", data=df)
        sns.stripplot(x="abs_"+_metric, y="id", data=df, color="black", alpha=0.3)
        if "angle" in _metric:
            plt.xlim([-0.5,90.5])
            plt.axvline(5, ls='dashed')
            plt.axvline(10)
        elif "iou" in _metric:
            plt.xlim([-0.05,1.05])
        elif "dist" in _metric:
            plt.axvline(0.10, ls='dashed')
            plt.axvline(0.15)

        plt.show()


def visualize_eval(data_metrics, args):
    pdata = defaultdict(list)
    if args.gt_floor:
        floor = "GTFLOOR"
    elif args.ransac:
        floor = "RANSAC"
    else:
        floor = Path(args.floor_roomdir).stem
    if args.gt_bed:
        bedroom = "GTBED"
    elif args.single_pass:
        bedroom = "single"
    else:
        bedroom = Path(args.bed_roomdir).stem
    imgdir = Path(args.outputdir) / f"{floor}--{bedroom}--{'templates' if args.use_templates else 'square'}_{args.method}{'_mc' if args.uncertain else ''}{int(round(args.percentile)) if args.uncertain else ''}_{args.postfix}"

    save_img = True
    try:
        imgdir.mkdir(parents=True)
    except:
        save_img = False
        print("dir already exists")

    with open(imgdir/"data.pickle", 'wb') as fp:
        pickle.dump(dict(data_metrics), fp, protocol=pickle.HIGHEST_PROTOCOL)

    for k, rooms in data_metrics["room"].items():
        kk = [k]*len(rooms)
        pdata["house"] += kk
        pdata["id"] += [x+"_"+y for x,y in zip(kk, rooms)]
        for _metric in {*data_metrics.keys()}-{"room", "pred_cfg", "gt_cfg", "pcd"}:
            pdata[_metric] += data_metrics[_metric][k]
            _data = np.abs(data_metrics[_metric][k])
            if "angle" in _metric:
                over90 = _data>90
                _data[over90] = -_data[over90]+180

            pdata["abs_"+_metric] += list(_data)

    df = pd.DataFrame.from_dict(pdata)
    df = df.sort_values(['id'])
    df.to_csv(imgdir/"data.csv")

    for _metric in {*data_metrics.keys()}-{"room", "pred_cfg", "gt_cfg", "pcd"}:
        plt.figure("abs_"+_metric,[10,15])
        sns.boxplot(x="abs_"+_metric, y="id", data=df)
        sns.stripplot(x="abs_"+_metric, y="id", data=df, color="black", alpha=0.3)
        if "angle" in _metric:
            plt.xlim([-0.5,90.5])
            plt.axvline(5, ls='dashed')
            plt.axvline(10)
        elif "iou" in _metric:
            plt.xlim([-0.05,1.05])
        elif "dist" in _metric:
            plt.axvline(0.10, ls='dashed')
            plt.axvline(0.15)
        if save_img:
            plt.savefig(imgdir/f"boxplot_{_metric}.png", bbox_inches="tight")
    return df


def get_means(x, args):
    try:
        print(x)
        df = pd.read_csv(str(x/"data.csv"))
        houses = {*df["house"]}
        tot_mean = 0
        for house in sorted(houses):
            mean = np.mean(df["abs_angle"][df["house"] == house])
            print(f"{house: <20}: {mean:.2f}")
            tot_mean += mean
        print(f"Total mean = {tot_mean/len(houses):.2f}")
    except Exception as e:
        print(e)
        return

def get_means_from_df(df, column="abs_angle", fx=None, verbose=True):
    houses = {*df["house"]}
    tot_mean = 0
    for house in sorted(houses):
        if fx is None:
            mean = np.mean(df[column][df["house"] == house])
        else:
            mean = np.mean(fx( df[column[0]][df["house"] == house], df[column[1]][df["house"] == house] ))
        if verbose:
            print(f"{house: <20}: {mean:.4f}")
        tot_mean += mean
    if verbose:
        print(f"Total mean = {tot_mean/len(houses):.4f}")
    return tot_mean/len(houses)

@torch.no_grad()
def main(args):
    args.use_templates = not args.no_templates
    floorpath = list(Path(args.floor_roomdir).iterdir())
    bedpath = list(Path(args.bed_roomdir).iterdir())

    floors = sorted(x.stem for x in floorpath)
    beds = sorted(x.stem for x in bedpath)
    datapath = Path(args.datapath)
    dataset = DataSelection(datapath.parent, default=datapath.name, notebook=False)

    if {*floors} != {*beds} or len(beds) != len(floors):
        raise ValueError("Rooms are not equal for both models")

    data_metrics = ddict(lambda: ddict(list))
    try:
        for floor_roomdir, bed_roomdir in zip(
            [x for x in sorted(floorpath) if x.stem in beds], sorted(bedpath)
        ):
            # print(floor_roomdir.stem)

            try:
                if args.gt_floor:
                    floor_triple = None, None, None
                else:
                    floor_triple = predict.init(floor_roomdir)
                if args.gt_bed:
                    bed_triple = None, None, None
                else:
                    bed_triple = predict.init(bed_roomdir, kill=args.kill)
            except:
                continue
            if args.use_templates:
                templates = get_templates(floor_roomdir, bed_roomdir, dataset)
            else:
                templates = {}
            all_samplepaths = list(dataset.iter(floor_roomdir.stem, sample=args.sample))
            batched_samplepaths = []
            for i, samplepaths in enumerate(all_samplepaths):
                batched_samplepaths.append(samplepaths)
                if i==len(all_samplepaths)-1 or len(batched_samplepaths)==args.batch_size:
                    pipeline(floor_roomdir, batched_samplepaths, floor_triple, bed_triple,
                             data_metrics, templates, args)
                    batched_samplepaths = []
    except KeyboardInterrupt:
        pass
    finally:
        df = visualize_eval(data_metrics, args)
        for col in ["abs_iou", "abs_angle_bed", "abs_segiou"]:
            get_means_from_df(df, column=col)


def create_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='Create dataset from Kaspard annotations')
    parser.add_argument("--gt_bed", action="store_true")
    parser.add_argument("--gt_floor", default=False, action="store_true")
    parser.add_argument("--no_templates", action="store_true")
    parser.add_argument("--method", default="icp")
    parser.add_argument("--single_pass", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--postfix", default="")
    parser.add_argument("--ransac", action="store_true")
    parser.add_argument("--save_toml", action="store_true")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--uncertain", default=False, action="store_true")
    parser.add_argument("--no_model", default=False, action="store_true")
    parser.add_argument("--skip_loc", default=False, action="store_true")
    parser.add_argument("--percentile", default=90, type=float)
    parser.add_argument("--kill", default=None, nargs='*')
    parser.add_argument("outputdir")
    parser.add_argument("floor_roomdir")
    parser.add_argument("bed_roomdir")
    parser.add_argument("datapath")
    return parser
