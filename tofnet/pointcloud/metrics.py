import numpy as np
from numpy.linalg import norm
from shapely.geometry import Polygon
from copy import deepcopy
import open3d

from tofnet.pointcloud.utils import rotate, transform_with_conf
from tofnet.pointcloud.visualize import visualize_pointcloud
from tofnet.annotations.segmentation import find_segmentation, find_z

def _rotate_from_cfg(cfg, vec):
    angles = (
        180-cfg["camera"]["inclination"], cfg["camera"]["lateral_inclination"]
    )
    alpha, beta = (np.radians(a) for a in angles)
    vec = rotate(alpha, vec, axis='x')
    vec = rotate(np.pi, vec, axis='z')
    vec = rotate(beta,  vec, axis='y')
    return vec

def floor_similarity(ground_cfg, pred_cfg, eps=1.e-8):
    """Uses cosine similarity to compare camera configs.

    Arguments:
        ground_cfg: ground truth config containing a "camera" section with
                    inclination and lateral_inclination
        pred_cfg: prediction (cfr ground_cfg)
        eps: epsilon value for cosine similarity metric

    Returns:
        similarity: cosine similarity
    """
    normal = np.array([1,1,1])
    ground_vec = _rotate_from_cfg(ground_cfg, normal)
    pred_vec = _rotate_from_cfg(pred_cfg, normal)
    similarity = np.degrees(np.arccos(np.clip(np.dot(ground_vec, pred_vec) / max(norm(ground_vec)*norm(pred_vec), eps), -1,1)))
    return similarity

def bed_similarity(ground_cfg, pred_cfg):
    """Compute 2d IoU for the bed, with common camera config."""
    raise NotImplementedError()

def get_bbox_points(cfg):
    l, w = cfg["bed"]["length"]/2, cfg["bed"]["width"]/2
    points = np.array([[l,w],[l,-w],[-l,-w],[-l,w],[l,w]])
    return points

def bprojIoU(gt_cfg, pred_cfg):
    """ Computes the bounding box IoU from 2 different configs while accounting for
        differences in floor rotation
    """
    from shapely import geometry
    res = []
    for cfg in [gt_cfg, pred_cfg]:
        height = cfg["camera"].get("height", 2.6)
        angles = (
            180-cfg["camera"]["inclination"], cfg["camera"]["lateral_inclination"],
            cfg["bed"]["orientation"]
        )
        alpha, beta, gamma = (np.radians(a) for a in angles)
        center = [cfg["bed"]["centerX"], cfg["bed"]["centerY"]]
        points = get_bbox_points(cfg)

        points = rotate(-gamma, points)
        points += center

        points3d = np.zeros((len(points), 3))
        points3d[:,:2] = points
        points3d[:,-1] = points3d[:,-1]-height
        points3d = rotate(-beta,  points3d, axis='y')
        points3d = rotate(-np.pi, points3d, axis='z')
        points3d = rotate(-alpha, points3d, axis='x')
        res.append(points3d)

    gt_points, pred_points = res
    # compute floor normal
    N = np.cross(gt_points[1,:]-gt_points[0,:], gt_points[3,:]-gt_points[0,:])
    d = N@gt_points[0,:]

    # project pred on that plane
    T = (d - pred_points@N)/(N@N)
    projected_pred_points = pred_points + np.expand_dims(T, axis=-1)*np.expand_dims(N, axis=0)

    # compute iou
    gt_poly = geometry.Polygon(gt_points[:,:2])
    pr_poly = geometry.Polygon(projected_pred_points[:,:2])
    iou = gt_poly.intersection(pr_poly).area/gt_poly.union(pr_poly).area
    return iou


def cIoU(gt_mask, pred_mask, c):
    """ Computes the image-wise pixel-based IoU
    """
    pok = pred_mask==c
    gok = gt_mask==c
    tp = np.sum(gok & pok)
    fp = np.sum((~gok) & pok)
    fn = np.sum(gok & (~pok))
    return tp/(tp+fn+fp)


def full_similarity(pcd, gt_cfg, pred_cfg, bed_class=2, sample=None, use_z=False):
    """Compute point-set IoU to have a complete view of bbox similarities."""
    gt_pcd = np.nan_to_num(deepcopy(pcd)["points"])
    pred_pcd = np.nan_to_num(deepcopy(pcd)["points"])

    # Pixel IoU
    ## Project pred_bed on pred_floor -> pred_rect
    shape = pcd["shape"][::-1]
    gt_pcd = transform_with_conf(gt_pcd, gt_cfg, shape)
    pred_pcd = transform_with_conf(pred_pcd, pred_cfg, shape)

    if use_z:
        gt_z = min(find_z(gt_pcd, end=1.4)+0.2, 1.35)
        mask_z = min(find_z(pred_pcd, end=1.4)+0.2, 1.35)
    else:
        gt_z, mask_z = 1.35, 1.35

    ## Annotate pixels -> compare -> IoU
    gt_mask = find_segmentation(gt_pcd, gt_cfg["bed"]["width"], gt_cfg["bed"]["length"], z=gt_z)
    pred_mask = find_segmentation(pred_pcd, pred_cfg["bed"]["width"], pred_cfg["bed"]["length"], z=mask_z)

    ## Rotation cossim
    cossim = np.degrees(np.arccos(np.cos(np.radians(
        gt_cfg["bed"]["orientation"] - pred_cfg["bed"]["orientation"]
    ))))
    err_center = np.sqrt(
        (gt_cfg["bed"]["centerX"] - pred_cfg["bed"]["centerX"])**2 +
        (gt_cfg["bed"]["centerY"] - pred_cfg["bed"]["centerY"])**2
    )
    err_len = gt_cfg["bed"]["length"] - pred_cfg["bed"]["length"]
    err_width = gt_cfg["bed"]["width"] - pred_cfg["bed"]["width"]
    divers_errors = (cossim, err_center, err_len, err_width)
    iou_proj = bprojIoU(gt_cfg, pred_cfg)

    pnts = pred_pcd[pred_mask==bed_class]
    pnts = np.mean(np.linalg.norm(pnts, axis=-1))

    return (0, iou_proj), divers_errors, pred_mask, gt_mask, pnts