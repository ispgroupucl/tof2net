import numpy as np
from tofnet.utils import pointcloud
from tofnet.utils.io import read_sample

"""
    ####################################
                PCD Utils
    ####################################
"""
def closest(pcd, to=(0,0,0)):
    to=tuple(to)
    pcd = pcd.copy()
    # pcd[(pcd[:,:,-1]<0.3)|(pcd[:,:,-1]>0.9)] = np.nan
    pcdz = pcd[:,:,:-1]
    toz = to[:-1]
    norm = np.linalg.norm(pcd-to, axis=-1)
    pcdz = pcd[:,:,:-1]
    normz = np.linalg.norm(pcdz-toz, axis=-1)
    closest = np.unravel_index(np.nanargmin(norm), norm.shape)
    return closest, pcd[closest]-to

def crop_pcd(pcd, width, length):
    bbox = pcd[:,:,1]>-width/2
    bbox &= pcd[:,:,1]<width/2
    bbox &= pcd[:,:,0]>-length/2
    bbox &= pcd[:,:,0]<length/2
    pcd = pcd.copy()
    pcd[~bbox] = np.nan
    return pcd
    
def find_z(pcd, width, length, bin_size=0.05):
    bbox = pcd[:,:,1]>-width/2
    bbox &= pcd[:,:,1]<width/2
    bbox &= pcd[:,:,0]>-length/2
    bbox &= pcd[:,:,0]<length/2
    pcd = pcd[bbox]
    hist,bin_edges = np.histogram(pcd[:,-1], bins=np.arange(0,2,bin_size))
    return bin_edges[np.argmax(hist)]+bin_size/2

def find_keypoints(pcd, width, length, points=None):
    points = points or [(0,0), # center
              (0,1),(1,0),(0,-1),(-1,0), # left, top, right, bottom
              (1,1),(-1,-1),(1,-1),(-1,1), # corners
              (0,1/2),(0,-1/2),(1/2,0),(-1/2,0) # smaller bbox
             ]
    z = find_z(pcd, width, length)
    pcd = pcd # crop_pcd(pcd, width, length)
    points = points * np.array([length/2, width/2, z])
    keypoints = []
    distances = []
    visible = []
    for point in points:
        keypoint, dist = closest(pcd, to=point)
        keypoints.append(keypoint)
        distances.append(dist)
        visible.append(np.all(dist<[0.1,0.1,0.2]))
    return keypoints, points, visible

"""
    ####################################
              Generate Heatmaps
    ####################################
"""
PIF_ZONES = {"gauss2d", "dist2d_const", "gauss3d", "dist3d", "dist3d_const"}
def pif_zone(
    zone_type, pcd=None, kp=None, kp_ind=None,  img_shape=None,
    sigma=0, sigma_ind=0, radius=0, radius_ind=0
):
    """
        This function computes everything pif-related
        The arguments are used on a case by case basis
    """
    if "2d" in zone_type:
        kpj, kpi = kp_ind
        j, i = np.indices(img_shape)
        if zone_type == "gauss2d":
            intensity = np.exp(- (((i-kpi)**2+(j-kpj)**2)**2) / (2*sigma_ind**2))
        elif zone_type == "dist2d_const":
            distances = (i-kpi)**2+(j-kpj)**2
            intensity = np.zeros(distances.shape)
            intensity[distances <= radius_ind**2] = 1
    else:
        filtered = np.isnan(pcd)
        pcd = np.nan_to_num(pcd, copy=True)
        ci, cj, ck = kp
        if zone_type == "gauss3d":
            intensity = np.exp(- (((pcd[:,0]-ci)**2+(pcd[:,1]-cj)**2+(pcd[:,2]-ck)**2)**2) / (2*sigma**2))
        elif zone_type == "dist3d":
            intensity = np.log(np.sqrt(((pcd[:,0]-ci)**2+(pcd[:,1]-cj)**2+(pcd[:,2]-ck)**2))+1e-7)
            max_int = np.nanmax(intensity)
            intensity = 1.0 - intensity/max_int
        elif zone_type == "dist3d_const":
            distances = ((pcd[:,0]-ci)**2+(pcd[:,1]-cj)**2+(pcd[:,2]-ck)**2)
            intensity = np.zeros(distances.shape)
            intensity[distances <= radius**2] = 1
        else:
            raise ValueError(f"Zone type {zone_type} is not supported")
        intensity[filtered[:,0]] = 0  # Should be enough since NaN is on all 3 dims
        intensity = np.reshape(intensity, img_shape[:2])
    return intensity


PAF_ZONES = {"cyl_dist2d_const", "cyl_gauss2d", "cyl_gauss3d"}
def paf_zone(
    zone_type, pcd=None, kp_list=None, kp_ind_list=None,  img_shape=None,
    sigma=0, sigma_ind=0, radius=0, radius_ind=0
):
    if "2d" in zone_type:
        j, i = np.indices(img_shape)
        if zone_type == "cyl_dist2d_const":
            distances = 0
            for kpj, kpi in kp_ind_list:
                distances = distances + (i-kpi)**2+(j-kpj)**2
            distances = distances / 4
            intensity = np.zeros(distances.shape)
            intensity[distances <= radius_ind**2] = 1
        elif zone_type == "cyl_gauss2d":
            intensity = 0
            for kpj, kpi in kp_ind_list:
                intensity = intensity + np.sqrt(((i-kpi))**2+((j-kpj))**2)
            intensity = np.exp(-((intensity)**2) / (2*sigma_ind**2))
            intensity = intensity / np.max(intensity)
    elif "3d" in zone_type:
        filtered = np.isnan(pcd)
        pcd = np.nan_to_num(pcd, copy=True)
        if zone_type == "cyl_gauss3d":
            intensity = 0
            for ci, cj, ck in kp_list:
                intensity = intensity + np.sqrt((pcd[:,0]-ci)**2+(pcd[:,1]-cj)**2+(pcd[:,2]-ck)**2)
            intensity = np.exp(-((intensity)**2) / (2*sigma**2))
            intensity = intensity / np.max(intensity)

        intensity[filtered[:,0]] = 0  # Should be enough since NaN is on all 3 dims
        intensity = np.reshape(intensity, img_shape[:2])

        
    return intensity

def generate(
    keypoint_dir, file_id, sample_names, 
    keypoint_grid, skeleton, style
):
    sample = read_sample(sample_names)
    outputfile = keypoint_dir / (file_id + ".npz")
    pcd = pointcloud.transform_with_conf(sample["pcd"]["points"], sample["conf"],  sample["image"].shape)
    width = sample["conf"]["bed"]["width"]
    length = sample["conf"]["bed"]["length"]
    keypoints, points, visible = find_keypoints(
        pcd, width, length, points=keypoint_grid
    )
    heatmaps = []
    if style in PIF_ZONES:
        for keypoint, vis, point in zip(keypoints, visible, points):
            if "3d" in style or vis: # do NOT show invisible ones
                heatmaps.append(pif_zone(
                    zone_type=style, pcd=pcd.reshape(-1,3),
                    kp=point, kp_ind=keypoint, img_shape=sample["image"].shape,
                    sigma=0.2, sigma_ind=40, radius=0.5, radius_ind=5
                )[None,...])
            else:
                heatmaps.append(np.zeros((1,)+sample["image"].shape))
    elif style in PAF_ZONES:
        for p1, p2 in skeleton:
            if "3d" in style or (visible[p1] and visible[p2]):
                heatmaps.append(paf_zone(
                    zone_type=style, pcd=pcd.reshape(-1,3),
                    kp_list=[points[p1], points[p2]],
                    kp_ind_list=[keypoints[p1], keypoints[p2]],
                    img_shape=sample["image"].shape,
                    sigma=0.3, sigma_ind=20, radius=0.5, radius_ind=5
                )[None,...])
            else:
                heatmaps.append(np.zeros((1,)+sample["image"].shape))

    else:
        raise ValueError(f"Zone type {style} is not supported")

    heatmaps = np.concatenate(heatmaps)
    heatmaps = np.transpose(heatmaps,(1,2,0))
    np.savez_compressed(outputfile, heatmaps)

def get_n_outputs(style):
    if style in PIF_ZONES:
        return 8
    elif style in PAF_ZONES:
        return 12

