import numpy as np
from tofnet.utils.pointcloud import clip_outliers
from tofnet.utils.io import read_sample
import cv2
import os

def generate_sample(style, sample):
    pci = sample["pcd"]["intensity"].copy()

    pci = clip_outliers(pci)
    pci = cv2.normalize(pci, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pci = cv2.flip(pci, -1)
    pci = pci.astype(np.uint8)
    img = cv2.equalizeHist(pci)

    shape = list(reversed(sample["pcd"]["shape"]))
    img = np.reshape(img, shape)
    return img

def generate(img_dir, file_id, sample_names, style):
    sample = read_sample(sample_names)
    out_file = img_dir / (file_id + ".png")
    img = generate_sample(style, sample)
    os.makedirs(out_file.parent, exist_ok=True)
    cv2.imwrite(str(out_file), img)

def inverse_names(iformat):
    """ identity function"""
    return iformat