import numpy as np

from tofnet.utils.io import read_sample
from tofnet.pointcloud.utils import transform_with_conf, extract_template
from tofnet.annotations.segmentation import find_segmentation
from tofnet.pointcloud.visualize import visualize_pointcloud
import matplotlib.pyplot as plt
from copy import deepcopy

def generate(template_dir, file_id, sample_names, style):
    if style is None:
        sample = read_sample(sample_names)
        template = extract_template(sample["pcd"]["points"], sample["conf"])
        outputfile = template_dir / (file_id + ".npz")
        np.savez_compressed(outputfile, template)
    elif style == "raw":
        sample = read_sample(sample_names)
        points = sample["pcd"]["points"]
        shape = sample["pcd"]["shape"][::-1]
        pcd = transform_with_conf(sample["pcd"]["points"], sample["conf"], shape)
        width = sample["conf"]["bed"]["width"]
        length = sample["conf"]["bed"]["length"]
        mask = find_segmentation(pcd, width, length)
        plt.imshow(mask)
        plt.show()
        mask = np.rot90(mask, 2).reshape(-1)
        points = transform_with_conf(deepcopy(points), sample["conf"], do_bed_transform=False)
        points = points[mask==2]
        visualize_pointcloud(points)
        template = extract_template(sample["pcd"]["points"], sample["conf"], False)
        outputfile = template_dir / (file_id + ".npz")
        np.savez_compressed(outputfile, template)
    else:
        raise ValueError("style must be None")