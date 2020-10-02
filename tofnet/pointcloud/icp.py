import open3d
from subprocess import check_output, CalledProcessError
from pypcd import pypcd
from tempfile import NamedTemporaryFile
from pathlib import Path
from pandas import read_csv
from io import StringIO
from contextlib import ExitStack
import numpy as np
import copy
from multiprocessing import Pool
from itertools import repeat, product
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy import spatial

from open3d.open3d.geometry import PointCloud

from tofnet.pointcloud.utils import rotate, angle, transform_with_conf, find_z
from tofnet.pointcloud.visualize import visualize_pointcloud, create_lineset

default_templates = [
    [1.8, 0.9],
    [2.0, 1.0],
    [2.0, 0.9],
    [2.1, 1.0],
    [1.8, 0.8],
]

def get_template(xsize=1.8, ysize=0.9, npoints=2000):
    xtempl = (np.random.beta(1.5, 1.5, size=npoints)-0.5)*xsize
    # ytempl = np.random.uniform(low=-ysize/2, high=ysize/2, size=target.shape[0])
    ytempl = (np.random.beta(1.4, 1.4, size=npoints)-0.5)*ysize
    template = np.c_[ # concatenate on last axis
        xtempl,
        ytempl,
        np.zeros(npoints)
    ]
    return template

def get_xytheta(transformation):
    x_trans = transformation[0,-1]
    y_trans = transformation[1,-1]
    theta = np.arctan2(transformation[1,0], transformation[0,0]) # np.arcsin(-transformation[0,1])
    transformation2 = np.linalg.inv(transformation)
    theta2 = -theta # np.arcsin(-transformation2[0,1])
    x, y = rotate(theta2, np.array([x_trans, y_trans]))
    return x, y, -theta

def init_transform(source):
    xinit = np.mean(source[:,0])
    yinit = np.mean(source[:,1])
    source = source - [xinit, yinit, 0]
    svd = np.linalg.svd(source[:,:-1])
    normal = svd[2][0]
    theta = np.arctan2(normal[0], normal[1]) # = angle([0,1], normal)
    source = rotate(theta, source, axis="z").copy()
    return source, xinit, yinit, theta


def create_config(x, y, theta, length=2.0, width=1.0, fitness=1.0):
    config = {
        "bed": {
            "centerX": x,
            "centerY": y,
            "orientation": theta,
            "length": length,
            "width": width, # Change this...
            "fitness":fitness
        }
    }
    return config


def grid_iou(bed_pcd, cfg, return_pcd=True, voxel_size=0.05, no_transform=False):
    """ Computes the fitness of a model around a segmentation

    Arguments:
        bed_pcd (np.ndarray): the 3d points segmented as being part of the bed
        cfg (dict): the proposed configuration
        return_pcd (bool): wether to return the pcd too
        voxel_size (float): size of the voxel to compensate for camera resolution
        no_transform (bool): wether to apply tranformation from the cfg on bed_pcd

    Returns:
        The fitness and the bed_pcd transformed if requested
    """
    cfg = cfg["bed"]
    if not no_transform:
        bed_pcd = bed_pcd.copy() - [cfg["centerX"], cfg["centerY"], 0]
        bed_pcd = rotate(np.radians(cfg["orientation"]), bed_pcd, axis="z")

    bed_pcd = np.round(bed_pcd[:,:2]/voxel_size)
    bed_pcd = np.unique(bed_pcd, axis=0)
    bed_pcd = bed_pcd*voxel_size

    x_ok = (bed_pcd[:,1] <= cfg["width"]/2) & (bed_pcd[:,1] >= -cfg["width"]/2)
    y_ok = (bed_pcd[:,0] <= cfg["length"]/2) & (bed_pcd[:,0] >= -cfg["length"]/2)
    n_ok = np.sum(x_ok & y_ok)
    n_elem = bed_pcd.shape[0]
    n_grid = (cfg["length"]+voxel_size)*(cfg["width"]+voxel_size)/(voxel_size*voxel_size)
    iou = n_ok/(n_grid+n_elem-n_ok)

    if return_pcd:
        return iou, bed_pcd
    else:
        return iou

from numba import njit
@njit
def multi_grid_iou(orig_bed_pcd, cfgs, voxel_size=0.05):
    """ Numba accelerated computation of the fitness of a model around a segmentation
        for multiple configs

    Arguments:
        orig_bed_pcd (np.ndarray): the 3d points segmented as being part of the bed
        cfgs (list): list of the possible configurations
        voxel_size (float): size of the voxel to compensate for camera resolution

    Returns:
        A list with the fitness for each cfg in cfgs
    """
    ious = np.zeros(len(cfgs))
    for i in range(cfgs.shape[0]):
        cfg = cfgs[i]
        bed_pcd = orig_bed_pcd - cfg[:2]
        theta = np.radians(cfg[2])
        rotmat = np.array([
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta),  np.cos(theta)],
            ])
        bed_pcd = bed_pcd@rotmat

        bed_pcd = np.round_(bed_pcd/voxel_size, 0, np.zeros_like(bed_pcd))
        bed_pcd_summed = np.unique(1024*bed_pcd[:,0]+bed_pcd[:,1])
        bed_pcd_0 = np.round_(bed_pcd_summed/1024, 0, np.zeros_like(bed_pcd_summed))
        bed_pcd_1 = bed_pcd_summed-bed_pcd_0*1024
        x_ok = (bed_pcd_1*2 <= int(cfg[4]/voxel_size)) & (bed_pcd_1*2 >= int(-cfg[4]/voxel_size))
        y_ok = (bed_pcd_0*2 <= int(cfg[3]/voxel_size)) & (bed_pcd_0*2 >= int(-cfg[3]/voxel_size))
        n_ok = np.sum(x_ok & y_ok)
        n_elem = len(bed_pcd_summed)
        n_grid = (cfg[3]+voxel_size)*(cfg[4]+voxel_size)/(voxel_size*voxel_size)
        ious[i] = n_ok/(n_grid+n_elem-n_ok)
    return ious


def dummy(source, templates=None, home=None):
    return create_config(0,0,0)


def no_icp(source, templates, home=None, use_2d=False):
    """ Computes the best-fitting bounding box around the source pixels.
        the bed model is estimated as being 2x1. The transformation is initialized with
        SVD, then a brute-force search around the initialization state tries to find a
        better fit
    """
    if source.shape[0] == 0:
        return create_config(0,0,0, fitness=0)
    origorig = source.copy()
    source, xinit, yinit, theta = init_transform(source)

    remove_out = False
    if remove_out:
        iq_range=0.5
        pcnt = (1 - iq_range) / 2
        xqlow, xqhigh = np.quantile(source[:,0], [pcnt, 1-pcnt])
        yqlow, yqhigh = np.quantile(source[:,1], [pcnt, 1-pcnt])
        xiqr, yiqr = xqhigh-xqlow, yqhigh-yqlow
        no_out = (source[:,0]<xqhigh+1.5*xiqr) & (source[:,0]>xqlow-1.5*xiqr) &\
                    (source[:,1]<yqhigh+1.5*yiqr) & (source[:,1]>yqlow-1.5*yiqr)
        source = source[no_out,:]

    remove_low = False
    if remove_low:
        not_low = (source[:,-1]>0.15) & (source[:,-1]<1.4)
        source = source[not_low,:]

    if source.shape[0] == 0:
        return create_config(0,0,0,fitness=0)


    # Recenter the data independently from point density
    xmin, xmax = np.min(source[:,0]), np.max(source[:,0])
    ymin, ymax = np.min(source[:,1]), np.max(source[:,1])
    x_center, y_center = xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2
    center2 = rotate(-theta, np.array([[x_center, y_center, 0]]), axis="z")
    xinit += center2[0][0]
    yinit += center2[0][1]
    source = source - [x_center, y_center, 0]

    angle_diffs = [-5.,-2.,-1.,0.,1.,2.,5.]
    center_diffs = [-0.2,-0.15,-0.10,-0.07,-0.05,-0.01,0,0.01,0.05,0.07,0.10,0.15,0.2]
    lengths = [2.,2.1]
    widths = [1.]
    trials = list(product(center_diffs, center_diffs, angle_diffs, lengths, widths))

    trialmap = np.array([np.array([xinit, yinit, np.degrees(theta)-90, 0.0, 0.0])+np.array(t) for t in trials])
    results = multi_grid_iou(origorig[:, :2], trialmap, 0.05)
    best = np.argmax(results)
    cfg = create_config(*trialmap[best])
    cfg["bed"]["fitness"] = np.max(results)
    return cfg


