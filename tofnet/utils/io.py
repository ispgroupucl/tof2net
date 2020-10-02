import os
import cv2
import numpy as np
from configparser import ConfigParser
from tofnet.utils.pointcloud import pypcd2xyz
from pypcd import pypcd
import qtoml as toml
from pathlib import Path

def write_sample(sample, samplefiles):
    """ Write back sample data to disk according to samplefiles paths

    Arguments:
        sample (dict): a dictionary of np.ndarrays and dict
        samplefiles (dict): a dictionary of Path-objects corresponding to their respective
                            data in the sample dict
    """
    for part in sample:
        filepath = Path(samplefiles[part])
        part_type = part.split('_')[0]
        if part_type in {"image", "mask", "xyz"}:
            cv2.imwrite(str(filepath), sample[part])
        elif part_type in {"conf"}:
            with filepath.open('w') as fp:
                toml.dump(sample[part], fp)
        else:
            raise ValueError(f"{part} is either not a valid type or not yet implemented.")


def read_sample(sample=None, grayscale=True, **kwargs):
    """ Reads the desired filepaths from disk according to their extensions

    Arguments:
        sample (dict):
        grayscale (bool): wether to read images as grayscale or not (= color)
        kwargs (dict): kwargs are used to overwrite or add to sample

    Returns:
        A dict representing the sample with same keys as sample and the objects loaded
        according to their respective data types

    .. note::

        The following will return the same sample::

            paths = {
                'image': Path('data/dataset/labeled/image/0.png'),
                'depth_xyz': Path('data/dataset/labeled/depth_xyz/0.npz')
            }
            read_sample(paths)

        Or ::

            paths = {
                'image': Path('data/dataset/labeled/image/0.png')
            }
            read_sample(paths, depth_xyz=Path('data/dataset/labeled/depth_xyz/0.npz'))

        Or ::

            im_path = Path('data/dataset/labeled/image/0.png')
            read_sample(paths, image=im_path, depth_xyz=Path('data/dataset/labeled/depth_xyz/0.npz'))
    """
    if sample is None:
        sample = kwargs
    else:
        sample.update(kwargs)
    result = {}
    for part in sample:
        filepath = sample[part]
        part_type = part.split('_')[0]
        if part_type in {"image", "mask"}:
            read_type = cv2.IMREAD_COLOR if (not grayscale) and part_type=="image" else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(str(filepath), read_type)
            result[part] = img
        elif part_type in {"xyz"}:
            read_type = cv2.IMREAD_COLOR
            img = cv2.imread(str(filepath), read_type)
            result[part] = img
        elif part_type in {"pcd"}:
            ppcd = pypcd.PointCloud.from_path(filepath)
            result[part] = pypcd2xyz(ppcd)
        elif part_type in {"depth", "keypoints", "template", "class", "var"}:
            with np.load(filepath) as npz_file:
                result[part] = npz_file.get('arr_0', npz_file.get("test")) # supports only one-array npz files
        elif part_type in {"conf"}:
            with filepath.open() as fp:
                result[part] = toml.load(fp)
        elif part_type in {"kaspardconf"}:
            confparser = ConfigParser()
            confparser.optionxform = lambda option: option # disable lower-casing
            confparser.read(filepath)
            result[part] = {s:dict(confparser.items(s)) for s in confparser.sections()}
            for skey, section in result[part].items():
                for key, item in section.items():
                    try:
                        result[part][skey][key] = float(item)
                    except:
                        pass
        else:
            raise ValueError(f"{part} is either not a valid type or not yet implemented.")
    return result

def get_datadir():
    if os.getenv("TOFNET_DATADIR"):
        return Path(os.getenv("TOFNET_DATADIR"))
    elif os.getenv("LOCALSCRATCH"):
        return Path(os.getenv("LOCALSCRATCH")) / "data"
    else:
        try:
            with open("dir.txt") as fp:
                data_dir = Path(fp.readlines()[0].strip())
        except :
            # print("Could not read dir.txt, reading default data/ folder")
            data_dir = Path("data")

        return data_dir

if __name__ == "__main__":
    print(get_datadir())