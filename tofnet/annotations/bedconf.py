import numpy as np

from tofnet.utils.io import read_sample

def compute_gt(config):
    x = config["bed"]["centerX"]
    y = config["bed"]["centerY"]
    cth = np.cos(np.radians(config["bed"]["orientation"]))
    sth = np.sin(np.radians(config["bed"]["orientation"]))
    length = config["bed"]["length"]
    width = config["bed"]["width"]
    return np.array([x,y,cth,sth,length,width])


def generate(conf_dir, file_id, sample_names, style):
    if style is None:
        sample = read_sample(sample_names)
        conf = compute_gt(sample["conf"])
        outputfile = conf_dir / (file_id + ".npz")
        np.savez_compressed(outputfile, conf)
    else:
        raise ValueError("style must be None")