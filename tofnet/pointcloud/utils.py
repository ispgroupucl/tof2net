"""Utility functions for (N,3) pointclouds."""
import numpy as np
from numba import jit
import numba as nb

def project_mat(pnt, norm):
    """Projects pnt following norm."""
    t = -(pnt@norm)/(norm.T@norm)
    ret = pnt + t
    return ret

def angle(a, b):
    """Computes angle between two vectors."""
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(a))
    return np.arccos(np.clip(c, -1, 1))


def project(pnt, norm):
    """Projects a point following a norm."""
    t = -np.sum(pnt*norm)/np.sum(norm*norm)
    ret = pnt+norm*t
    return ret/np.linalg.norm(ret)

def _gauss1d(x, σ, μ=0):
    return (1/σ*np.sqrt(2*np.pi))*np.exp(-0.5*((x-μ)/σ)**2)


def mollify(x, sigma=0.01):
    return np.convolve(x, _gauss1d(x, sigma))


def sum_of_gaussians(x0, y0, fwhm=0.01):
    return weighted_sum_of_gaussians(x0,y0, np.ones_like(x0))

def weighted_sum_of_gaussians(x0, y0, z0, fwhm=0.01):
    x0 = x0[:, None, None]
    y0 = y0[:, None, None]
    z0 = z0[:, None, None]
    log2 = np.log(2)
    def _gauss(x, y):
        x = x[None, ...]
        y = y[None, ...]
        return np.sum(z0*np.exp(-4* log2 * ((x-x0)**2 + (y-y0)**2) / fwhm**2), axis=0)

    return _gauss


def rotval(*args, **kwargs):
    return rotate(*args, **kwargs)

def rotate_mesh(theta, x, y):
    rot_mat = {
        "x": np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
    }["x"]

    return np.einsum('ji, mni -> jmn', rot_mat, np.dstack([x, y]))



def rotate(theta, vec, axis="x"):
    """Gets rotation matrix from angle and applies it."""
    if vec.shape[-1]==3:
        rot_mat = {
            "x": np.array([
                [1,0,0,],
                [0,np.cos(theta), -np.sin(theta)],
                [0,np.sin(theta),  np.cos(theta)],
            ]),
            "y": np.array([
                [np.cos(theta),0, np.sin(theta)],
                [0,1,0,],
                [-np.sin(theta),0,  np.cos(theta)],
            ]),
            "z": np.array([
                [np.cos(theta), -np.sin(theta),0],
                [np.sin(theta),  np.cos(theta),0],
                [0,0,1],
            ]),
        }[axis]
    elif vec.shape[-1]==2:
        rot_mat = {
            "x": np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)],
            ])
        }[axis]
    else:
        raise NotImplementedError("only works for 2d or 3d rotation")
    vec = np.array(vec)
    return (rot_mat@vec.T).T


def crop_pcd(pcd, width, length, z=None):
    bbox = pcd[:,1]>-width/2
    bbox &= pcd[:,1]<width/2
    bbox &= pcd[:,0]>-length/2
    bbox &= pcd[:,0]<length/2
    if z is not None:
        bbox &= pcd[:,-1]<z
    return pcd[bbox]


def find_z(pcd, bin_size=0.05, start=0.4, end=1.5):
    if len(pcd.shape) == 3:
        pcd = pcd.reshape(-1, 3)
    hist,bin_edges = np.histogram(pcd[:,-1].reshape(-1), bins=np.arange(start,end,bin_size))
    return bin_edges[np.argmax(hist)]+bin_size/2


def extract_template(pcd, cfg, do_bed_transform=True):
    pcd = transform_with_conf(pcd, cfg, shape=None, do_bed_transform=do_bed_transform)
    width = cfg["bed"]["width"]
    length = cfg["bed"]["length"]
    z = find_z(pcd)
    bed = crop_pcd(pcd, width, length, z)
    return bed


def transform_with_conf(pcd, cfg, shape=None, do_bed_transform=True):
    """ Rotates a pcd's 3d coordinates based on a configuration

    Arguments:
        pcd (np.ndarray): the 3d coordinates of the pointcloud
        cfg (dict): the camera and bed configuration
        shape (tuple): if set, the pcd is reset to the shape
        do_bed_transform (bool): wether to orient the pcd according to the bed
                                thus centering the 3d points on the bed

    Returns:
        The pcd rotated according to the cfg
    """
    try:
        height = cfg["camera"].get("height", 2.6)
    except KeyError:
        raise ValueError("config doesn't contain camera")
    angles = (
        180-cfg["camera"]["inclination"], cfg["camera"]["lateral_inclination"],
        cfg["bed"]["orientation"] if do_bed_transform else 0
    )
    alpha, beta, gamma = (np.radians(a) for a in angles)
    pcd = rotate(alpha, pcd, axis='x')
    pcd = rotate(np.pi, pcd, axis='z')
    pcd = rotate(beta,  pcd, axis='y')
    pcd[:,-1] = pcd[:,-1]+height

    if do_bed_transform:
        # Rotate Around Center of bed
        center = np.array([[cfg["bed"]["centerX"], cfg["bed"]["centerY"], 0]])
        pcd = pcd - center
        pcd = rotate(gamma, pcd, axis='z')

    # to 3-channel x,y,z "image"
    if shape is not None:
        pcd = np.reshape(pcd, shape+(3,))
        pcd = np.rot90(pcd, 2)

    return pcd


@jit(nb.f8[:,::1](nb.f8, nb.f8, nb.f8[:,::1]), nopython=True)
def rotate_(alpha, beta, pcd):
    alpha = np.array([
        [1.,0.,0.,],
        [0., np.cos(alpha), np.sin(alpha)],
        [0.,-np.sin(alpha), np.cos(alpha)],
    ])
    gamma = np.array([
        [-1, 0,0.],
        [0, -1,0.],
        [0, 0,1.],
    ])
    beta = np.array([
        [np.cos(beta),0., -np.sin(beta)],
        [0.,1.,0.,],
        [np.sin(beta),0.,  np.cos(beta)],
    ])
    pcd = pcd@(alpha@gamma@beta)
    return pcd

def fast_twconf(pcd, cfg, shape=None): # about 10-15x faster
    """ A faster numba accelerated version of :func:`transform_with_conf`
    """
    try:
        height = cfg["camera"].get("height", 2.6)
    except KeyError:
        raise ValueError("config doesn't contain camera")
    angles = (
        180-cfg["camera"]["inclination"], cfg["camera"]["lateral_inclination"]
    )
    alpha, beta = (np.radians(a) for a in angles)
    pcd = pcd.astype(np.float64, order='C')
    pcd = rotate_(alpha, beta, pcd)
    pcd[:,-1] = pcd[:,-1]+height
    # to 3-channel x,y,z "image"
    if shape is not None:
        pcd = np.reshape(pcd, shape+(3,))
        pcd = np.rot90(pcd, 2)

    return pcd