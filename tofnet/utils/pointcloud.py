"""Pointcloud utility functions."""

import os
import cv2
import numpy as np
import open3d
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from tofnet.data.generators import DEFAULT_SEGMENT_COLORS
from tofnet.pointcloud.utils import *
from tofnet.pointcloud.regressors import SVDRegressor
from tofnet.pointcloud.utils import rotate_ as fast_rotate
from tofnet.pointcloud.icp import no_icp
from scipy.interpolate import griddata


class PointCloudConfig:
    """From input pointcloud, get configuration for camera and bed.

    Attributes:
        pointcloud (np.array): Point cloud as x,y,z, (intensity, segmentation)
        segmentation (np.array): segmentation information of shape (N,)
        floor_normal (np.array): (3,) array pointing upwards from floor
        config (dict) : final configuration
    """
    def __init__(self, pointcloud):
        self.pointcloud = pointcloud
        self.config = {}


    @property
    def shape(self):
        return self.pointcloud["shape"]

    @property
    def ishape(self):
        s = self.pointcloud["shape"]
        return (s[1], s[0])

    def find_floor(self, segmentation, floor_id=1, threshold=0.1,
                   min_samples=6):
        """Find floor centroid and normal vector to the floor plane.

        Arguments:
            segmentation (np.array): array of shape (N,) or (H,W) and dtype (u)int(8)
            floor_id (int): index of floor segmentation
            threshold: threshold (in meters) for RANSAC regression
            min_samples: minimum #samples used for one RANSAC iteration

        Returns:
            centroid: floor centroid of shape (3,)
            floor_normal: normal to the floor. shape (3,)

        """
        pointcloud = self.pointcloud["points"]
        segmentation = segmentation.cpu()
        if len(segmentation.shape) >= 2:
            segmentation = np.rot90(segmentation, 2).reshape(-1)
        floor = pointcloud[segmentation == floor_id, :]
        floor = floor[~np.isnan(floor[:, 0])]
        centroid = np.mean(floor, axis=0)
        floor = floor - centroid # center floor in space

        # Compute RANSAC to choose best points
        ransac_model = SVDRegressor()
        ransac = RANSACRegressor(
            ransac_model, residual_threshold=threshold,
            min_samples=min_samples
        )
        ransac.fit(floor, np.zeros(floor.shape[0]))
        floor_inlier = floor[ransac.inlier_mask_]#  + centroid
        # print(f"better mean ? {np.mean(floor_inlier[:,-1])}")

        # Compute SVD for normal computation
        svd = np.linalg.svd(floor_inlier)
        normal = svd[2][-1]

        if normal[1] < 0:
            normal = -normal
        color = np.zeros(floor.shape[0], dtype=np.uint8)
        color[ransac.inlier_mask_]
        self.floor_normal = normal
        dist_scores = np.abs(floor@normal)
        fitness = (
            floor.shape[0],
            np.sum(ransac.inlier_mask_),
            np.sum(ransac.inlier_mask_)/len(ransac.inlier_mask_),
            np.mean(dist_scores), np.var(dist_scores),
        )
        return centroid, normal, fitness # ransac.estimator_.norm_.reshape(3)

    def rotate_gravity(self, normal, centroid):
        """Rotate pointcloud according to floor segmentation.

        Arguments:
            normal (np.ndarray): A 3d vector indicating the floor's normal orientation
            centroid (np.ndarray): A 3d point indicating the floor's center

        Returns:
            pointcloud (np.ndarray): The rotated 3d pointcloud
            config (dict): The config-like representation of the floor-oriented calibration
        """

        pointcloud = self.pointcloud["points"]
        base_axes = np.array([[1,0,0], [0,1,0], [0,0,1]])

        # -- Compute pitch
        nx  = normal.copy()
        nx[0] = 0

        xangle = 2*np.pi/2 - angle(base_axes[-1], nx) # rotate around x
        nxprojz = rotval(np.pi-xangle, normal, axis="x")
        rotcentroid = rotval(np.pi-xangle, centroid, axis="x")

        zangle = np.pi
        nxprojz = rotate(zangle, nxprojz, axis="z")
        rotcentroid = rotate(zangle, rotcentroid, axis="z")

        # -- Compute roll
        nxprojz[1] = 0 # should already be close to zero, maybe assert ?
        yangle =  -np.pi/2 + angle(base_axes[0], nxprojz) # rotate around y
        nxproj_orig = rotval(yangle, nxprojz, axis="y")
        rotcentroid = rotval(yangle, rotcentroid, axis="y")


        # -- Apply on pointcloud
        pointcloud = fast_rotate(np.pi-xangle, yangle, pointcloud.astype(np.float64, order='C'))

        config = {
            "camera": {
                "inclination": np.degrees(xangle),
                "lateral_inclination": np.degrees(yangle),
                "height": -rotcentroid[2]
            }
        }
        self.pointcloud["points"] = pointcloud - [0,0,rotcentroid[2]]
        return pointcloud, config

    def find_bed(self, segmentation, templates=None, bed_id=2, method="icp", home=""):
        """Find bed centroid and normal vector

        Args:
            segmentation (np.array): array of shape (N,) or (H,W) and dtype (u)int(8)
                with N 1st dimension of self.pointcloud["points"]
            bed_id (int): index of bed segmentation

        Returns:
            centroid: floor centroid of shape (3,)
            normal: normal indicating the length direction of shape (3,)

        """
        segmentation = segmentation.cpu()
        if len(segmentation.shape) >= 2:
            segmentation = np.rot90(segmentation, 2).reshape(-1)
        if np.sum(segmentation==bed_id) == 0:
            print("No bed pixels found")
            return None
        # visualize_pointcloud(self.pointcloud["points"])
        bed = self.pointcloud["points"][segmentation==bed_id]
        bed = bed[~np.isnan(bed[:,0])]
        z = find_z(bed, start=0.2)
        bed = bed[(bed[:,-1]>(z-0.3))]

        # ICP point set registration
        methods = {
            "no_icp": no_icp,
        }

        config = methods[method](bed, templates=templates, home=home)
        return config


def clip_outliers(x, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, qhigh = np.quantile(x[~np.isnan(x)], [pcnt, 1-pcnt])
    iqr = qhigh - qlow
    return np.clip(x, qlow-1.5*iqr, qhigh+1.5*iqr)

def save_image(file, pcd):
    """ Saves an intensity image based on a 3d pointcloud

    Arguments:
        file (Path): path where to save the image
        pcd (dict): Pointcloud with all the information, dict with different fields
                    and their respective np.ndarray
    """
    pci = pcd["intensity"].reshape(pcd["shape"][1], pcd["shape"][0])
    pci = clip_outliers(pci)
    pci = cv2.normalize(pci, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pci = cv2.flip(pci, -1)
    pci = pci.astype(np.uint8)
    pci = cv2.equalizeHist(pci)
    os.makedirs(file.parent, exist_ok=True)
    if np.count_nonzero(pci)==0:
        raise ValueError("image is black")
    cv2.imwrite(str(file), pci)

def save_depth(file, pcd):
    """ Saves an depth image based on a 3d pointcloud

    Arguments:
        file (Path): path where to save the image
        pcd (dict): Pointcloud with all the information, dict with different fields
                    and their respective np.ndarray
    """
    pcd = np.linalg.norm(pcd["points"], axis=-1).reshape(pcd["shape"][::-1])

    pcd = cv2.normalize(pcd, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pcd = cv2.flip(pcd, -1)
    pcd = pcd.astype(np.uint8)
    # DONT DO HISTOGRAM EQUALIZATION, I think.
    os.makedirs(file.parent, exist_ok=True)
    if np.count_nonzero(pcd)==0:
        raise ValueError("depth is black")
    cv2.imwrite(str(file), pcd)

def save_xyz(file, pcd):
    """ Saves an xyz image based on a 3d pointcloud

    Arguments:
        file (Path): path where to save the image
        pcd (dict): pointcloud with all the information, dict with different fields
                    and their respective np.ndarray
    """
    pcd = pcd["points"].reshape(pcd["shape"][1], pcd["shape"][0],3)
    pcd = pcd.copy()
    for i in range(3):
        pcd[:,:,i] = cv2.normalize(pcd[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pcd = cv2.flip(pcd, -1)
    pcd = pcd.astype(np.uint8)
    os.makedirs(file.parent, exist_ok=True)
    if np.count_nonzero(pcd)==0:
        raise ValueError("xyz is black")
    cv2.imwrite(str(file), pcd)

def pypcd2xyz(ppcd):
    """ Reads a pypcd and converts it to a manageable python structure

    Arguments:
        ppcd (PyPcd): raw pcd structure as read from a binary file

    Returns:
        pcd (dict): a more easy to manitpulate pcd data-structure in dict-form,
                    with the following keys: 'points' (np.ndarray) for the 3d points,
                    'shape' (tuple) for the original recorded shape
    """
    from numpy.lib.recfunctions import repack_fields
    res = repack_fields(ppcd.pc_data[["x","y","z"]])
    points = res.view(np.float32).reshape(res.shape + (-1,))
    pcd = {"points": points}
    for field in ppcd.fields:
        pcd[field] = ppcd.pc_data[field]
    metadata = ppcd.get_metadata()
    pcd["shape"] = (metadata['width'], metadata['height'])
    return pcd


def _interpolate_fx(_mat):
    i, j = np.indices(_mat.shape)
    for method in ["linear", "nearest"]:
        ok = ~np.isnan(_mat)
        _mat = griddata(
            (j[ok], i[ok]), _mat[ok], (j, i), method=method
        )
    return _mat

def interpolate_nan(mat):
    """ Interpolates a 3d matrix """
    if len(mat.shape) == 3:
        nmat = []
        for ic in range(3):
            nmat.append(_interpolate_fx(mat[:,:,ic]))
        return np.stack(nmat, axis=-1)
    elif len(mat.shape) == 2:
        return _interpolate_fx(mat)
    else:
        raise ValueError(f"Matrix of shape: {mat.shape} not supported")


if __name__ == "__main__":
    pass