import os
import cv2
import numpy as np
from tofnet.utils import pointcloud
from tofnet.utils.io import read_sample
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tofnet.pointcloud.utils import find_z


def segment(sample):
    shape = sample["pcd"]["shape"][::-1]
    conf = sample["conf"]
    conf["bed"] = conf["bed"][0]
    pcd = pointcloud.transform_with_conf(sample["pcd"]["points"], conf, shape)
    width = conf["bed"]["width"]
    length = conf["bed"]["length"]

    mask = find_segmentation(pcd, width, length)
    # save_segmentation(out_file, mask)
    return mask

def find_segmentation(pcd, width, length, z=None):
    mask = np.zeros(pcd.shape[:-1])
    if z is None:
        pcdcrop = crop_pcd(pcd, width, length)
        z = find_z(pcdcrop)
    bbox = bbox_area(pcd, width, length, z+0.15)
    floor = pcd[:,:,-1]<0.15
    mask[floor] = 1
    mask[bbox] = 2
    return mask

def save_segmentation(out_file, mask):
    os.makedirs(out_file.parent, exist_ok=True)
    cv2.imwrite(str(out_file), mask)

def bbox_area(pcd, width, length, z=None):
    bbox = pcd[:,:,1]>-width/2
    bbox &= pcd[:,:,1]<width/2
    bbox &= pcd[:,:,0]>-length/2
    bbox &= pcd[:,:,0]<length/2
    if z is not None:
        bbox &= pcd[:,:,-1]<z
    return bbox

def crop_pcd(pcd, width, length, z=None):
    bbox = bbox_area(pcd, width, length, z)
    pcd = pcd.copy()
    pcd[~bbox] = np.nan
    return pcd


def generate(mask_dir, file_id, sample_names, style):
    sample = read_sample(sample_names)
    out_file = mask_dir / (file_id + ".png")
    mask = generate_sample(style, sample)
    save_segmentation(out_file, mask)

def generate_sample(style, sample):
    if "interpolated" in style:
        pcd = sample["pcd"]["points"]
        p_shape = pcd.shape
        pcd = pcd.reshape(sample["image"].shape+(3,))
        sample["pcd"]["points"] = pointcloud.interpolate_nan(pcd).reshape(p_shape)
    
    rstyle = style.rsplit('_', 1)[-1]
    if "normal" in rstyle:
        return segment(sample)
    else:
        raise ValueError(f"Style {style} is not supported")
