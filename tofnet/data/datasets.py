"""Dataset Representations of directories

Examples
--------
>>> import matplotlib.pyplot as plt
>>> ds = DirectoryDataset("data/cityscapes", (1,128,128))
>>> plt.imshow(ds[0])
>>> plt.show()
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from tofnet.data.generators import to_tensor
from tofnet.data.preprocess import crop_center
import numpy as np
from itertools import cycle
import bisect
from scipy.ndimage.morphology import binary_erosion
from skimage import measure
from copy import deepcopy
import importlib
from tofnet.utils.config import get_class
from tofnet.utils.io import read_sample, get_datadir
from tofnet.data import samplers



def parsepcd(pc):
    pci = pc.pc_data['intensity'].reshape(pc.height, pc.width)
    pci = cv2.normalize(pci, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pci = cv2.flip(pci, -1)
    pci = pci.astype(np.uint8)
    pci = cv2.equalizeHist(pci)
    pci = to_tensor(pci)
    pcd = ((pc.pc_data['x']+pc.pc_data['y']+pc.pc_data['z'])/3).reshape(pc.height, pc.width)
    pcd = cv2.normalize(pcd, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    pcd = cv2.flip(pcd, -1)
    pcd = pcd.astype(np.uint8)
    pcd = cv2.equalizeHist(pcd)
    pcd = to_tensor(pcd)
    return {"image": pci, "depth": pcd}


class DirectoryDataset(Dataset):
    """ Dataset representation of a typical segmentation database

    Arguments:
        root_dir (str): directory where all the data is stored in subdirectories
        dimensions (tuple): shape of the input after optional cropping
        dtypes (list): list of strings that correspond to subdirs in root_dir from which
                    the data should be loaded
        resize (str): method of resizing the input, 'later' when the dimensions need
                    to be kept, 'crop3' to divide image
        input_type (str): the input_type (from dtypes) should be used for visualization
        only_labeled (bool): wether to only load the labeled data or not
        ignore_index (int): the value of pixels to ignore for segmentation
        repeat (int): the amount of time each data should be repeated for 1 epoch
        classes (list): list of class-names
        read_dims (tuple): dimension to which the read images are immediately resized
    """
    def __init__(self, root_dir, dimensions, dtypes, resize="later", input_type="image",
                    only_labeled=True, ignore_index=255, repeat=1, classes=None,
                    read_dims=None, **kwargs):

        # Convert dtypes to string (FIXME: change dtypes to output_types?)
        dir_names = dtypes.copy()

        # Get all filenames
        self.input_type = input_type
        self.fnames = {}
        for i in range(len(dtypes)):
            fnames =  (Path(root_dir) / dir_names[i]).iterdir()
            self.fnames[dtypes[i]] = sorted(fnames)

        self.classes = classes
        self.load_labels = "mask" in dtypes
        self.only_labeled = only_labeled
        if self.load_labels:
            if self.only_labeled:
                label_names = set()
                for m in self.fnames["mask"]:
                    label_names.add(m.stem)
                for key in self.fnames:
                    if key != "mask":
                        self.fnames[key] = [elem for elem in self.fnames[key] if elem.stem in label_names]

        # Check if all dimension correspond
        all_len = [len(elem) for elem in self.fnames.values()]
        assert len(np.unique(all_len))==1, f"The #images is not equal for the different inputs:\n {dtypes}\n {all_len}"

        # Save other params
        self.grayscale = len(dimensions)<3 or (dimensions[0]==1)
        self.shape     = dimensions if len(dimensions)==3 else (1,)+dimensions
        self.resize    = resize
        self.multiplier = 3 if resize == "crop3" else 1
        self.repeat = repeat
        # self.multiplier *= self.repeat
        self.ignore_index = ignore_index
        self.length = all_len[0]*self.multiplier
        self.read_dims = read_dims
        self.fname_samples = [dict(zip(self.fnames, i)) for i in zip(*self.fnames.values())]
        # TODO: intelligent rescaling based on max rescaling & crop?

    def __len__(self):
        return self.length * self.repeat

    def __getitem__(self, real_idx):
        real_idx = real_idx % self.length
        idx = real_idx // self.multiplier
        part_idx = real_idx % self.multiplier

        resize_method = {
                    # Methods, Their Arguments
            "crop": lambda x: crop_center(x),
            "crop3": lambda x: crop_center(x, [-4,0,4][part_idx]),
            "later": lambda x: x,
            "resize": lambda x: x # TODO
        }[self.resize]
        result = {}
        result = read_sample(self.fname_samples[idx], grayscale=self.grayscale)
        fnames = self.fname_samples[idx]
        for key in fnames:
            img = result[key]
            key_type = key.split('_')[0] # FIXME
            if key_type in {"mask"} and img is None:
                # TODO: this won't work
                img = torch.ones(*self.shape, dtype=torch.uint8) * self.ignore_index
            elif key_type in {"class"}:
                img = torch.tensor(0) if img < 0.7 else torch.tensor(1) # FIXME do this already in dataset
            elif key in {"pcd"}:
                img = torch.tensor(img["points"])
                img[torch.isnan(img)] = 0
                img = img.reshape(*self.shape[1:], 3)
                img = torch.flip(img, dims=(0,1))
                img = img.reshape(-1, 3)
            else:
                interp = cv2.INTER_NEAREST if key_type in ["mask"] else  cv2.INTER_LINEAR
                if self.read_dims is not None:
                    img = cv2.resize(img, tuple(self.read_dims), interpolation=interp)
                img = to_tensor(img)
            if key_type in {"image", "depth", "keypoints", "pcd", "class", "var"}: # Add dimension to correspond to expected
                img = img.unsqueeze(0)
            elif key_type in {"mask"}: # Remove unwanted classes and merge classes
                new_num = 1
                num_dict = {"": 0}
                for i, cl in enumerate(self.classes):
                    if cl in num_dict:
                        img[img==i+1] = num_dict[cl]
                    else:
                        img[img==i+1] = new_num
                        num_dict[cl] = new_num
                        new_num += 1
            img = resize_method(img) # Apply resize
            result[key] = img
        return result


    def get_filenames(self, indices):
        if type(indices) == int:
            indices = [indices]
        fnames = [None]*len(indices)
        for i, idx in enumerate(indices):
            idx = idx % self.length
            part_idx = idx % self.multiplier
            postfix = "_%d" % part_idx if part_idx != 0 else ""
            filepath = self.fnames[self.input_type][idx//self.multiplier]
            fnames[i] = filepath.parent / (filepath.stem + postfix + filepath.suffix)
        return fnames

    @staticmethod
    def collate_fn(batch):
        result = {}
        for key in batch[0].keys():
            values = (elem[key] for elem in batch)
            values = tuple(values)
            result[key] = torch.cat(values)
        return result

class SamplingDataset(DirectoryDataset):
    """Samples dataset, either per regex group or entirely.

    Arguments:
        ratio (float) : ratio of sampling per group
        fixed_number (int) : images per group (mutex with ratio)
        groups : regex for group (i.e. : 'Fold\d+--(.*)_\d+')
        seed : random seed for reproducibility.
        weights : dictionary with multipliers for different groups

    .. note::

        See DirectoryDataset for other parameters.

    """
    def __init__(self, root_dir, dimensions, dtypes, resize="later", input_type="image",
                 ratio=0.1, fixed_number=-1, groups="", seed=456, weights=None, **kwargs):
        assert ratio <= 1.0, f"The ratio of selected images ({ratio}) should be <=1"
        super().__init__(root_dir, dimensions, dtypes, resize, input_type, **kwargs)

        fnames = self.fnames
        import random
        random.seed(seed)

        # Compute mask to avoid some data
        if groups=="":
            n_tot = self.length//self.multiplier
            n_samples = max(round(n_tot*ratio), 1) if fixed_number<1 else min(n_tot, fixed_number)
            indices = random.sample(range(n_tot), n_samples)
        else:
            import re
            from collections import defaultdict as ddict
            rr = re.compile(groups)
            stems = [x.stem for x in fnames[self.input_type]]
            grouped_fnames = ddict(list)
            for idx, fname in enumerate(fnames[self.input_type]):
                grp = rr.match(fname.stem).group(1)
                grouped_fnames[grp].append(idx)

            indices = []
            for grp, grp_ids in grouped_fnames.items():
                n_samples_grp = max(round(len(grp_ids)*ratio), 1) if fixed_number<1 else min(len(grp_ids), fixed_number)
                indices += random.sample(grp_ids, n_samples_grp)
            n_samples = len(indices)


        # Sample the files
        for dtype, files in fnames.items():
            if weights is None:
                self.fnames[dtype] = sorted(files[i] for i in indices)
            else:
                weighted_fnames = []
                for fname in [files[i] for i in indices]:
                    for prefix_name, weight in weights.items():
                        if fname.name.startswith(prefix_name):
                            weighted_fnames += weight*[fname]
                            break
                    else:
                        weighted_fnames.append(fname)
                self.fnames[dtype] = sorted(weighted_fnames)
                n_samples = len(weighted_fnames)

        # for dtype, files in fnames.items():
        #     fnames[dtype] = [files[i] for i in indices]
        self.fname_samples = [dict(zip(self.fnames, i)) for i in zip(*self.fnames.values())]
        self.length = n_samples*self.multiplier


class RotationDataset(DirectoryDataset):
    """Adds the 4 rotations to a Dataset"""
    def __init__(self, root_dir, dimensions, dtypes, resize="later", input_type="image", **kwargs):
        super().__init__(root_dir, dimensions, dtypes, resize, input_type, **kwargs)
        # self.multiplier *= 4 # 4 more rotations

    def get_filenames(self, indices):
        if type(indices) == int:
            indices = [indices]
        indices = [idx//4 for idx in indices]
        rots = [idx%4 for idx in indices]
        filenames = super().get_filenames(indices)
        assert len(rots)==len(filenames)
        rotated_fnames = []
        for rot, fname in zip(rots, filenames):
            rotated_fnames.append(fname.parent / f"{fname.stem}_r{rot}{fname.suffix}")
        return rotated_fnames

    def __len__(self):
        return 4*super().__len__()

    def __getitem__(self, idx):
        # TODO : rotate depth !!!!!!
        real_idx = idx // 4
        rot_idx = idx % 4
        result = super().__getitem__(real_idx)
        target = torch.Tensor([rot_idx])
        out = {"class": target}
        for dtype in result:
            out[dtype] = result[dtype].rot90(rot_idx, (-2, -1))
        return out


class DummyRotationDataset(DirectoryDataset):
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        out = {"class": torch.Tensor([0])}
        for dtype in result:
            out[dtype] = result[dtype]
        return out


class WeakPointsDataset(DirectoryDataset):
    def __init__(self, root_dir, dimensions, dtypes, resize="later", input_type="image", points=100, radius=5, ignore_index=255, **kwargs):
        self.points= points
        self.radius = radius if type(radius) is list else [radius]
        self.ignore_index = ignore_index
        super().__init__(root_dir, dimensions, dtypes, resize, input_type, ignore_index=ignore_index, **kwargs)


    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        np.random.seed(idx)
        mask = result.get("mask")
        if mask is None:
            return result
        mask = mask.numpy()
        n_classes = np.max(mask)
        new_mask = np.ones(mask.shape) * self.ignore_index
        for i, radius in zip(range(n_classes+1), cycle(self.radius)):
            _, rows, cols = np.where(mask==i)
            if rows.size<1:
                continue
            for _ in range(self.points):
                id0 = np.random.choice(rows.size)
                new_mask[0] = cv2.circle(new_mask[0], (cols[id0], rows[id0]), radius=radius, color=i, thickness=-1)
        result["mask"] = torch.from_numpy(new_mask)
        return result

class WeakLinesDataset(DirectoryDataset):
    """Dataset replacing accurate masks with lines

    Args:
        root_dir: directory containing the images
        dimensions: final dimensions for images
        load_labels:
        resize:
        only_labeled:
        lines:
        linesize:
        ignore_index:
        from_edges:

    """

    def __init__(self, root_dir:str, dimensions:tuple, dtypes:list, resize:str="later", input_type="image", lines:int=4, linesize:int=2, ignore_index:int=255,from_edges:bool=False, **kwargs):
        self.lines = lines
        self.linesize = linesize if type(linesize) is list else [linesize]
        self.from_edges = from_edges
        self.ignore_index = ignore_index
        super().__init__(root_dir, dimensions, dtypes, resize, dtypes, input_type, **kwargs)

    def color_instances(self, img):
        """img is semantic segmentation"""
        labels, num = measure.label(img, return_num=True)
        return labels, num

    def edges(self, img):
        """return edge of binary image

        >>> img = np.zeros((5,5), dtype=np.uint8)
        >>> img[1:5,1:5] = 1
        >>> img
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1],
               [0, 1, 1, 1, 1],
               [0, 1, 1, 1, 1],
               [0, 1, 1, 1, 1]], dtype=uint8)
        >>> edges(img)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1],
               [0, 1, 0, 0, 1],
               [0, 1, 0, 0, 1],
               [0, 1, 1, 1, 1]], dtype=uint8)

        """
        return img.astype(np.uint8) - binary_erosion(img)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        np.random.seed(idx)
        import matplotlib.pyplot as plt
        # img = result["image"]
        mask = result.get("mask")
        if mask is None:
            return result
        mask = mask.numpy()
        instances, num_instances = self.color_instances(mask)
        new_mask = np.ones(mask.shape, dtype=np.uint8) * self.ignore_index
        for i in range(1, num_instances+1):
            if self.from_edges:
                instance = self.edges(instances==i)
            else:
                instance = (instances==i).astype(np.uint8)
            _, rows, cols = np.nonzero(instance)
            class_num = mask[0, rows[0], cols[0]]
            _, no_rows, no_cols = np.where(mask!=class_num)
            if rows.size<3:
                new_mask[0, rows, cols] = class_num
                continue
            for _ in range(self.lines):
                id0, id1 = np.random.choice(rows.size, 2, replace=False)
                # yy, xx = line(rows[id0], cols[id0], rows[id1], cols[id1])
                yy, xx, _ = weighted_line(rows[id0], cols[id0], rows[id1], cols[id1], w=self.linesize[class_num%len(self.linesize)], rmax=self.shape[-1])
                tmp_mask = np.ones(mask.shape) * self.ignore_index
                tmp_mask[0,yy,xx] = class_num
                # tmp_mask[binary_dilation(tmp_mask==class_num, iterations=self.linesize[class_num%len(self.linesize)])] = class_num
                tmp_mask[0,no_rows,no_cols] = 0
                new_mask[tmp_mask==class_num] = class_num
        result["mask"] = torch.from_numpy(new_mask) # .to(dtype=torch.uint8)
        return result

def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

"""Source: https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays"""
def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getattr__(self, name):
        for dataset in self.datasets:
            try:
                return getattr(dataset, name)
            except AttributeError:
                continue
        raise AttributeError(f"'dataset' object has no '{name}' attribute")

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


# Aliases
Directory = Default = DefaultDataset = default = DirectoryDataset
Rotation = rotation = RotationDataset
WeakLines = weaklines = WeakLinesDataset

datasets = importlib.import_module(__name__)
data_dir = get_datadir()

def get_datasets(cfg):
    """ Returns the train/val/test datasets corresponding to the information in the cfg """
    base_path  = data_dir / cfg["dataset"]["name"]
    tr_path = base_path / "train"
    val_path = base_path / "val"
    te_img_path = base_path / "test"

    if "part" in cfg["dataset"]: # multi-part dataset from different directories
        parts = deepcopy(cfg["dataset"]["part"])
        train_dss = []
        val_dss = []
        test_dss = []
        for part in parts:
            train_dss.append(get_class({"root_dir":tr_path, **cfg["dataset"], **part}, datasets))
            # Order : val_dataset overwrites part overwrites dataset
            val_dss.append(get_class({"root_dir":val_path, **cfg["dataset"], **part, **cfg.get("val_dataset", {})}, datasets))
            test_dss.append(get_class({"root_dir":te_img_path, **cfg["dataset"], **part, **cfg.get("test_dataset", {}), "only_labeled":False}, datasets))

        train_ds = ConcatDataset(train_dss)
        print("length", len(train_ds))
        val_ds = ConcatDataset(val_dss)
        test_ds = ConcatDataset(test_dss)
        train_ds.collate_fn = val_ds.collate_fn = test_ds.collate_fn = DirectoryDataset.collate_fn
        sampler = get_class({"type": cfg["dataset"].get("sampler", "RandomSampler"), "data_source": train_ds}, samplers)
    else:
        # Make Dataset objects from paths
        train_ds = get_class({"root_dir":tr_path, **cfg["dataset"]}, datasets)
        val_ds   = get_class({"root_dir":val_path, **cfg["dataset"], **cfg.get("val_dataset", {})}, datasets)
        test_ds  = get_class({"root_dir":te_img_path, **cfg["dataset"], **cfg.get("test_dataset", {}), "only_labeled":False}, datasets)

        sampler = get_class({"type": cfg["dataset"].get("sampler", "RandomSampler"), "data_source": train_ds}, samplers)

    return train_ds, val_ds, test_ds, sampler
