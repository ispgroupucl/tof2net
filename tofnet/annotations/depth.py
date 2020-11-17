import numpy as np
from tofnet.utils.io import read_sample
from tofnet.utils import pointcloud
from tofnet.pointcloud.utils import transform_with_conf, fast_twconf
from sklearn.neighbors import KDTree, DistanceMetric
from numba import njit, prange

@njit(parallel=True)
def numba_compute_normals(n_elem, indices, distances, max_dist, pcd):
    normals = np.zeros(pcd.shape)
    for i in prange(n_elem):
        ind, dist = indices[i], distances[i]
        if np.sum(dist) == 0: # Nan value
            continue
        ind = ind[dist<=max_dist]
        if len(ind)<3: # Normal computation needs at least 3 points!
            continue
        points = pcd[ind]
        norm_vec = np.linalg.svd(points)[2][-1]

        # Check wether vector is facing inward instead of outwards
        if norm_vec[2] < 0:
            norm_vec = -norm_vec

        normals[i] = norm_vec
    return normals

def compute_normals(_pcd, height_encoded=True, max_dist=0.1, n_neighbors=9, n_iter=20):
    _zero_array = np.array([0.0,0.0,0.0])

    n_neighbors += 1 # It will always find itself too
    if len(_pcd.shape) == 3:
        n_elem = _pcd.shape[0]*_pcd.shape[1]
    else:
        n_elem = _pcd.shape[0]
    pcd = _pcd.copy()
    pcd = pcd.reshape(n_elem, 3)

    tree = KDTree(pcd, leaf_size=10)
    metric = DistanceMetric.get_metric('euclidean')

    distances, indices = tree.query(pcd, k=n_neighbors)
    normals = numba_compute_normals(n_elem, indices, distances, max_dist, pcd)
    normals = normals.reshape(_pcd.shape)
    return normals



def generate(depth_dir, file_id, sample_names, style):
    sample = read_sample(sample_names)
    outputfile = depth_dir / (file_id + ".npz")
    depth = generate_sample(style, sample)
    np.savez_compressed(outputfile, depth)

def generate_sample(style, sample):
    pcd = sample["pcd"]["points"]

    # Choose between interpolation or nan->0
    if "interpolated" in style:
        pcd = np.reshape(pcd, sample["image"].shape+(3,))
        pcd = pointcloud.interpolate_nan(pcd)
        pcd = np.reshape(pcd, (np.sum(sample["image"].shape), 3))
    else:
        pcd = np.nan_to_num(pcd)

    if "height" in style:
        pcd = fast_twconf(
            pcd.copy(), sample["conf"], sample["image"].shape,
        )
        if "xyz" in style:
            depth = pcd
        else:
            depth = pcd[:,:,-1]
    else:
        pcd = np.reshape(pcd, sample["image"].shape+(3,))
        if "distance" in style:
            pcd = np.sum(np.sqrt(pcd*pcd), axis=-1, keepdims=False)
        elif "xyz" not in style:
            raise ValueError(f"Style {style} is not supported")
        depth = np.rot90(pcd, 2, axes=(0,1)).copy()

    # After the rstyle transformation,
    # this will compute the normals for each point
    if "normals" in style:
        depth = compute_normals(depth, "height" in style)

    if "xyz" not in style:
        depth = np.expand_dims(depth, axis=-1)

    return depth

def get_n_outputs(style):
    return 3 if "xyz" in style else 1


def inverse_names(iformat):
    """ Inverses an input format name to
        generate the config used
    """
    iformat_split = iformat.split("_")
    if "depth" != iformat_split[0]:
        return None
    # Remove _<number>
    iformat = iformat if len(iformat_split) < 3 else "_".join(iformat_split[:-1])
    # Remove random seeds if any
    for style in  ["height", "distance", "xyz"]:
        if style in iformat:
            break
    else:
        return None
    ssplit = iformat.split(style)
    if len(ssplit)>2 and len(ssplit[-1])==6:
        iformat = iformat[:-6]
    sbase = iformat.split(style)[0]
    style = sbase+"_"+style if sbase[-1] != '_' else sbase+style
    return style

