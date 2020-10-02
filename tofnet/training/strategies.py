import inspect
import torch
import json
import toml
import collections
import types
import numpy as np
from collections import OrderedDict, defaultdict
from pathlib import Path
from shutil import rmtree, copy2
import os
from subprocess import check_output, DEVNULL
import tempfile
import pandas as pd
import shutil
from functools import lru_cache
from itertools import repeat
from multiprocessing import Pool
from filelock import FileLock

import tofnet.annotations.keypoints
import tofnet.annotations.depth
import tofnet.annotations.segmentation
import tofnet.annotations.template
import tofnet.annotations.image

from tofnet import annotations
from tofnet.utils.io import get_datadir

class Strategy:
    def clean(self):
        pass

    def __len__(self):
        return 1

    def __call__(self, config, iteration):
        raise NotImplementedError()

class Eval(Strategy):
    def modify_config(self, config):
        all_config = self.config
        for key, value in config.items():
            if hasattr(value, 'get') and "eval" in value:
                print(value["eval"])
                config[key] = eval(value["eval"])
            elif hasattr(value, 'get'):
                self.modify_config(value)
            elif hasattr(value, '__getitem__') and len(value) !=0 and hasattr(value[0], 'get'):
                for v in value:
                    self.modify_config(v)

    def __call__(self, config, iteration):
        self.config = config # access for eval... dirty but I like it ;)
        self.modify_config(config)
        return config

@lru_cache()
def call_command(*command):
    return check_output(command, stderr=DEVNULL).decode().strip()

class Git(Strategy):
    def get_branch(self):
        return call_command(*["git", "rev-parse", "--symbolic-full-name", "--abbrev-ref", "HEAD"])

    def get_diff(self):
        return call_command(*["git", "diff-files", "--patch"])

    def get_commit(self):
        return call_command(*["git", "rev-parse", "HEAD"])

    def get_modified(self):
        return call_command(*["git", "ls-files", "-m"]).split("\n")

    def __call__(self, config, iteration):
        config["git"] = {
            "branch": self.get_branch(),
            "commit": self.get_commit(),
            "modified": self.get_modified(),
            "diff": self.get_diff()
        }
        return config

class DatasetMaker(Strategy):
    """Creates new dataset from pcd and kaspard.conf or any other base files

    Arguments:
        style (str): base type to make (enum: gauss2d, gauss3d etc.)
        dtype (str): which type do you want to create (enum: keypoints, depth)
        format_type (str): wether to replace in input- or output-format
        pool_size (int): #threads to spawn for faster annotation
        annotation_file (str): which keypoint file to parse
        base_dtypes (list(str)): list of base dtypes which are needed to make
                                the new dataset. the 1st element will be
                                used for the NEW filenames

    .. note::
        Not idempotent (order matters). Put before LeaveOneOut to create once and for all,
        put after LOO to create each time. (should not pose a problem tx to random seed)

    """
    def __init__(
        self,*, style=None, dtype="keypoints", format_type="output",
        pool_size=12, annotation_file="annotations.json",
        base_dtypes=("image", "pcd", "conf"), use_seed=True
    ):
        self.style = style
        self.dtype = dtype
        self.format_type = format_type if "format" in format_type else f"{format_type}_format"
        self.base_dtypes = base_dtypes
        self.pool_size = pool_size
        self.generated = False
        self.annotation_file = annotation_file

        if style is not None:
            self.tmp_dir = Path(f"{dtype}_{''.join(style.split('_'))}")
        else:
            self.tmp_dir = Path(dtype)

        self.data_dir = get_datadir()
        self.made_dirs = []


    def get_names(self, directory, idx):
        result = {}
        for dtype in self.base_dtypes:
            dtype_dir = directory / dtype
            for val in dtype_dir.glob(f"{idx}.*"):
                result[dtype_dir.stem] = val
        return result

    def get_sample_ids(self, directory):
        for img in directory.glob(f"{self.base_dtypes[0]}/*"):
            yield img.stem

    # FIXME: Generate this automatically
    def generate_keypoints(self, kp_dir, data_dir, file_id):
        annotations.keypoints.generate(
            kp_dir, file_id, self.get_names(data_dir, file_id),
            self.keypoint_grid, self.skeleton, self.style
        )
        return

    def generate_depth(self, depth_dir, data_dir, file_id):
        annotations.depth.generate(
            depth_dir, file_id, self.get_names(data_dir, file_id), self.style
        )
        return

    def generate_mask(self, mask_dir, data_dir, file_id):
        annotations.segmentation.generate(
            mask_dir, file_id, self.get_names(data_dir, file_id), self.style
        )
        return

    def generate_image(self, img_dir, data_dir, file_id):
        annotations.image.generate(
            img_dir, file_id, self.get_names(data_dir, file_id), self.style
        )
        return

    def generate_template(self, template_dir, data_dir, file_id):
        annotations.template.generate(template_dir,
                                      file_id,
                                      self.get_names(data_dir, file_id),
                                      self.style
        )

        return

    def generate_bedconf(self, conf_dir, data_dir, file_id):
        annotations.bedconf.generate(conf_dir, file_id,
                                     self.get_names(data_dir, file_id),
                                     self.style)
        return

    def __call__(self, config, iteration):
        root_dir = self.data_dir / config["dataset"]["name"]
        with Pool(self.pool_size) as pool:
            for directory in root_dir.iterdir():
                if not (directory / f"{self.base_dtypes[0]}").exists():
                    continue # Avoid using a non-data containing directory

                new_data_dir = directory / self.tmp_dir
                lock = FileLock(f"{str(new_data_dir)}.dir.lock")
                with lock:
                    if os.path.exists(new_data_dir):
                        print(f"[ds-maker] {new_data_dir} already exists")
                        continue
                    print(f"[ds-maker] Creating {new_data_dir}")
                    os.makedirs(new_data_dir , exist_ok=False)  # Could fail in case of collision
                    # self.made_dirs.append(new_data_dir)

                    if self.dtype == "keypoints":
                        with open(Path(directory) / self.annotation_file) as annot_file:
                            annot = json.load(annot_file)
                        self.keypoint_grid = annot["categories"][2]["_keypoint_grid"]
                        self.keypoint_names = annot["categories"][2]["keypoints"]
                        self.skeleton = annot["categories"][2]["skeleton"]
                        pool_fx = self.generate_keypoints
                    elif self.dtype == "depth":
                        pool_fx = self.generate_depth
                    elif self.dtype == "mask":
                        pool_fx = self.generate_mask
                    elif self.dtype == "image":
                        pool_fx = self.generate_image
                    elif self.dtype == "template":
                        pool_fx = self.generate_template
                    elif self.dtype == "bedconf":
                        pool_fx = self.generate_bedconf
                    else:
                        raise ValueError(f"Dtype {self.dtype} not supported")
                    pool_args = zip(
                        repeat(new_data_dir),
                        repeat(directory),
                        self.get_sample_ids(directory)
                    )

                    pool.starmap(
                        pool_fx,
                        pool_args
                    )

        config["dataset"]["dtypes"] += [str(self.tmp_dir)]
        n_outputs = {
            "keypoints": annotations.keypoints.get_n_outputs,
            "depth": annotations.depth.get_n_outputs
        }.get(self.dtype, lambda x: None)(self.style)
        for i, out_name in enumerate(config["network"][self.format_type].copy()):
            if out_name == self.dtype:
                config["network"][self.format_type][i] = (
                    f"{str(self.tmp_dir)}_{n_outputs}"
                    if n_outputs is not None else #"output" in self.format_type and
                    f"{str(self.tmp_dir)}"
                )
                break
        else:
            raise ValueError(f"No {self.dtype} found in {self.format_type}")
        self.generated = True
        return config


class Sampler(Strategy):
    """Sample dataset in different parts using some kind of Dataset"""

    def __init__(self, *, sizes, act_time=None):
        self.sizes = sizes
        self.iterations = len(sizes)

    def __len__(self):
        return self.iterations

    def __call__(self, config, iteration):
        size = self.sizes[iteration]
        config["dataset"]["parent"] = ["SamplerDataset"]

def copy_file(src:Path, dest:Path):
    setname = dest.parents[1].name
    if src.suffix == ".npz":
        with np.load(src) as npzfile:
            array = npzfile.get(setname, npzfile.get("arr_0"))
            np.savez_compressed(dest, array)
    else:
        copy2(src, dest)

class LeaveOneOut(Strategy):
    """Leave-one-out/K-fold implementation.

    Arguments:
        prefix (list): a list of strings for the different fold's test-prefixes, at iteration k
                        prefixes[k] will be used for testing and k+1 for validation
        infix (str): the string around which to split in order to match the data's filename
                    with the gives prefixes

    """
    def __init__(self, *, prefixes, source_dir=None, unsupervised_prefixes=None,
                 target_dir="", act_time=None, infix="-", sample_size=np.inf,
                 weight_dir=None, train_is_val=False, seed=0):
        # TODO : get data dir in a better uniform way
        np.random.seed(seed)
        self.data_dir = get_datadir()
        self.prefixes = prefixes
        self.infix = infix
        self.act_time = act_time
        self.unsupervised_prefixes = unsupervised_prefixes or []
        self.target_dir =  Path(tempfile.mkdtemp(prefix="tmp_",dir=self.data_dir)) # self.data_dir / Path(target_dir)
        if source_dir is not None:
            self.source_dir = self.data_dir / Path(source_dir)
        else:
            self.source_dir = None
        self.iterations = len(self.prefixes)
        self.sample_size = sample_size
        self.train_is_val = train_is_val
        self.weight_dir = None if weight_dir is None else Path(weight_dir)
        self.sets = ["train", "val", "test"]
        for prefix in self.prefixes:
            if self.weight_dir is not None and not (self.weight_dir / prefix / "model.pt").exists():
                print("WARNING : Some rooms have no pretrained weights !")
                break


    def create_sample(self, test_prefix, val_prefix):
        dirs = self.dirs
        rmtree(self.target_dir, ignore_errors=True)
        dtypes = dirs["labeled"] # FIXME : only works because labeled is bigger

        for s in self.sets:
            for k in dtypes:
                os.makedirs(self.target_dir / s / k, exist_ok=True)
            for jfile in (self.source_dir / "labeled").glob("*.json"):
                copy2(jfile, self.target_dir / s / jfile.name)

        # Copy labels from real data
        idict = defaultdict(int)
        for kind, idirs in dirs.items():
            for idir in idirs:
                for image_file in (self.source_dir / kind / idir).glob("*"):
                    prefix = image_file.stem.rsplit(self.infix, 1)[0]
                    s = {test_prefix:'test', val_prefix:'val'}.get(prefix, 'train')
                    if self.train_is_val:
                        if s == "train":
                            continue
                        elif s=="val":
                            copy_file(image_file, self.target_dir / "train" / idir / image_file.name)
                            idict[prefix] += 1

                    if s == "train" and idict[prefix] >= self.sample_size:
                        continue

                    copy_file(image_file, self.target_dir / s / idir / image_file.name)
                    idict[prefix] += 1


    def __len__(self):
        return self.iterations

    def clean(self):
        rmtree(self.target_dir)

    def __call__(self, config, iteration):
        if self.source_dir is None:
            self.source_dir = self.data_dir / config["dataset"]["name"]
        self.dirs = {
            "labeled": config["dataset"]["dtypes"],
            "unlabeled": list(set(config["dataset"]["dtypes"])-{*config["network"]["output_format"]})
        }
        self.create_sample(self.prefixes[iteration], self.prefixes[(iteration+1)%self.iterations])
        config["saves"]["path"] = os.path.join(config["saves"]["path"], self.act_time)
        config["saves"]["act_time"] = self.prefixes[iteration]
        config["dataset"]["name"] = self.target_dir.name
        if self.weight_dir is not None and (self.weight_dir / self.prefixes[iteration] / "model.pt").exists():
            config["network"]["pretrained_weights"] = str(self.weight_dir / self.prefixes[iteration] / "model.pt")
        return config

def get_config(infile):
    infile = Path(infile)
    with open(infile, 'r') as fp:
        if infile.suffix == ".toml":
            return toml.loads(fp.read())
        else:
            return json.loads(fp.read(), object_pairs_hook=OrderedDict)


def sample_with(files, *sizes, f=np.random.permutation):
    sampled_files = list(f(files))
    return_files = []
    sizes += (len(sampled_files),)
    start = 0
    for size in sizes:
        return_files.append(sampled_files[start:start+size])
        start += size
    return return_files


class GridSearch(Strategy):
    """ Grid search using hyper key with list.

        Args:
            config: default config file.
            best_select (string or bool): "max" or "min"
            act_time (string): directory name (usually containing timestamp)

        Example Config:
        .. code-block:: toml

            [[strategy]]
            type = 'GridSearch'
            best_select = 'max'

            [training]
            batch_size.hyper = [4,8,16]

    """
    def __init__(self, *, config=None, best_select="max", act_time=None):
        self.all = {}
        self.iteration = -1

        # Parse config ones to initialize indices and know length
        self.act_time = act_time
        self.cnt = []
        self._modify_config(config)
        self._init_inidices()
        self.iteration = 0

        # Will store important info from config & which directories were created
        self.tmp_dirs = []
        self.tmp_metric = None
        self.max_select = best_select=="max"

    def __len__(self):
        return self.iterations

    def _modify_config(self, config, index=-1):
        for key, value in config.items():
            if isinstance(value, collections.Mapping) and "hyper" in value:
                if hasattr(value["hyper"], "__getitem__") or value["hyper"] == "list":
                    hyper_list = value.get("list") or value["hyper"]
                    if index >= 0:
                        val_index = self.indices[self.iteration][index]
                        config[key] = hyper_list[val_index]
                        index += 1
                    else:
                        self.cnt.append(len(hyper_list))
                else:
                    raise ValueError(value["hyper"]+" is not a possible value for hyperparams")
            elif isinstance(value, collections.Mapping):
                index = self._modify_config(value, index)
        return index

    def modify_config(self, config):
        self._modify_config(config, 0)

    def _init_inidices(self):
        def _rec_parsing(li):
            if len(li) == 0:
                return [[]]
            ret = _rec_parsing(li[1:])
            new_ret = []
            for partial in ret:
                for ind in range(li[0]):
                        new_ret.append(
                            [ind]+partial
                        )
            return new_ret
        self.indices = _rec_parsing(self.cnt)
        self.iterations = len(self.indices)

    def __call__(self, config, iteration):
        self.iteration = iteration

        # model saving in subdir!!
        if config["saves"]["act_time"] != self.act_time:
            self.act_time = config["saves"]["act_time"]
        tmp_dir = os.path.join(config["saves"]["path"], self.act_time, "tmp")
        self.tmp_dirs.append(os.path.join(tmp_dir, str(self.iteration)))
        config["saves"]["path"] = tmp_dir
        config["saves"]["act_time"] = str(self.iteration)
        self.tmp_metric = config["saves"]["monitor"]

        self.modify_config(config)
        return config

    def clean(self):
        if self.iteration == len(self)-1:
            # Select best & Clean up subdirectories
            best_value = -np.inf if self.max_select else (lambda x,y: x<y)
            best_dir = None
            for tmp in self.tmp_dirs:
                with open(os.path.join(tmp, "metrics.csv")) as fp:
                    df = pd.read_csv(fp)
                    new_val = df[self.tmp_metric].max()
                    if cmp_fx(new_val, best_value):
                        best_value = new_val
                        best_dir = tmp

            tmp = Path(best_dir)
            for fdname in os.listdir(best_dir):
                try:
                    shutil.move(os.path.join(best_dir, fdname), tmp.parents[1])
                except shutil.Error:
                    print(f"Could not move {fdname}")
            shutil.rmtree(tmp.parent)
            self.tmp_dirs = []


class RandomSplit(LeaveOneOut):
    def __init__(self, act_time=None, val_portion=0.1, test_portion=0., **kwargs):
        """ Randomly splits data in train/val/test according to the given portions

        Arguments:
            val_portion (float): portion of the data used for validation (between 0 and 1),
                                must be >0
            test_portion (float): portion of the data used for testing  (between 0 and 1),
                                if =0 the validation set is repeated for testing
        """
        super().__init__(prefixes=["nofold"], act_time=act_time, **kwargs)
        self.val_portion = val_portion
        self.test_portion = test_portion

    def __len__(self):
        return 1

    def choose_sampling(self, files):
        """
        Do special sampling from two sets of files
        """

        val_number = int(round(self.val_portion*len(files["train"])))
        test_number = int(round(self.test_portion*len(files["train"])))
        val_files, train_files = sample_with(
            files["train"], val_number + test_number
        )
        if test_number == 0:
            test_files = val_files
        else:
            val_files, test_files = sample_with(
                val_files, val_number
            )

        return dict(train=train_files, val=val_files, test=test_files)

    def create_sample(self, test_room, val_room):
        dirs = self.dirs["labeled"]
        kind = "labeled"
        rmtree(self.target_dir, ignore_errors=True)
        dtypes = dirs

        for s in self.sets:
            for k in dtypes:
                os.makedirs(self.target_dir / s / k, exist_ok=True)
            for jfile in (self.source_dir / "labeled").glob("*.json"):
                copy2(jfile, self.target_dir / s / jfile.name)

        sampling = None
        for i, idir in enumerate(dirs):
            # get file suffix
            suffix = next((self.source_dir / kind / idir).iterdir()).suffix

            # Split file in train/val/test (done only once)
            if sampling is None:
                files = defaultdict(list)
                for image_file in sorted((self.source_dir / kind / idir).iterdir()):
                    files["train"].append(image_file)

                sampling = self.choose_sampling(files)

            # Copy images to tmp dir
            for set_name, set_list in sampling.items():
                for image_file in set_list:
                    source_file = (self.source_dir / kind / idir / f"{image_file.stem}{suffix}")
                    copy_file(source_file,
                            self.target_dir / set_name / idir / source_file.name
                    )

    def __call__(self, config, iteration):
        config = super().__call__(config, iteration)
        return config


class Specialize(LeaveOneOut):
    def __init__(self, train_size=1, val_size=1, act_time=None, prefix="--", val_portion=0.,
                       weight=1, **kwargs):
        super().__init__(act_time=act_time, **kwargs)
        self.train_size = train_size
        self.val_size = val_size
        self.val_portion = val_portion
        self.prefix = prefix
        self.weight = weight

    def choose_sampling(self, files):
        """
        Do special sampling from two sets of files
        TODO : refactor this disgusting code
        """
        tr1, val1, test_files = sample_with(
            files["test"],
            self.train_size, self.val_size
        )
        val_number = int(round(self.val_portion*len(files["train"])))
        val2, tr2 = sample_with(
            files["train"],
            val_number
        )
        train_files = tr1 + tr2
        val_files = val1 + val2
        return dict(train=train_files, val=val_files, test=test_files)

    def create_sample(self, test_room, val_room):
        dirs = self.dirs["labeled"]
        kind = "labeled"
        rmtree(self.target_dir, ignore_errors=True)
        dtypes = dirs
        test_prefix = test_room.rsplit(self.prefix, 1)[0]

        for s in self.sets:
            for k in dtypes:
                os.makedirs(self.target_dir / s / k, exist_ok=True)
            for jfile in (self.source_dir / "labeled").glob("*.json"):
                copy2(jfile, self.target_dir / s / jfile.name)


        for i, idir in enumerate(dirs):
            # get file suffix
            suffix = next((self.source_dir / kind / idir).iterdir()).suffix
            if i==0:
                files = defaultdict(list)
                for image_file in sorted((self.source_dir / kind / idir).iterdir()):
                    room = image_file.stem.rsplit(self.infix, 1)[0]
                    prefix = image_file.stem.rsplit(self.prefix, 1)[0]
                    if room == test_room:
                        files["test"].append(image_file)
                    elif prefix != test_prefix:
                        files["train"].append(image_file)
                    else:
                        files["unused"].append(image_file)
                sampling = self.choose_sampling(files)
            for set_name, set_list in sampling.items():
                for image_file in set_list:
                    source_file = (self.source_dir / kind / idir / f"{image_file.stem}{suffix}")
                    copy_file(source_file,
                            self.target_dir / set_name / idir / source_file.name
                    )

    def __call__(self, config, iteration):
        config = super().__call__(config, iteration)
        config["dataset"]["weights"] = {self.prefixes[iteration]:self.weight}
        home_name = self.prefixes[iteration].rsplit(self.prefix)[0]
        if self.weight_dir is not None:
            model_paths = list(self.weight_dir.glob(f"{home_name}*/model.pt"))
            if len(model_paths)==1:
                config["network"]["pretrained_weights"] = str(model_paths[0])

        return config


class Finetune(LeaveOneOut):
    def __init__(self, train_size=1, val_size=1, act_time=None, prefix="--", weight=1, **kwargs):
        super().__init__(act_time=act_time, **kwargs)
        self.train_size = train_size
        self.val_size = val_size
        self.prefix = prefix
        self.weight = weight
        if self.weight_dir is None or not (self.weight_dir / "base_config.toml"):
            print("[WARNING] need weight_dir containing base_config.toml")
        else:
            self.base_dirs = {d.name:d for d in self.weight_dir.iterdir() if d.is_dir()}

    def create_sample(self, test_prefix, val_prefix):
        dirs = self.dirs["labeled"]
        rmtree(self.target_dir, ignore_errors=True)
        kind = "labeled"
        dtypes = dirs

        for s in self.sets:
            for k in dtypes:
                os.makedirs(self.target_dir / s / k, exist_ok=True)
            for jfile in (self.source_dir / "labeled").glob("*.json"):
                copy2(jfile, self.target_dir / s / jfile.name)

        for i, idir in enumerate(dirs):
            # get file suffix
            suffix = next((self.source_dir / kind / idir).iterdir()).suffix
            if i==0:
                files = list((self.source_dir / kind / idir).glob(
                            f"{test_prefix}{self.infix}*"
                        ))
                train, val, test = sample_with(files,
                                                self.train_size, self.val_size
                )

            for set_name, set_list in dict(train=train, val=val, test=test).items():
                for image_file in set_list:
                    source_file = (self.source_dir / kind / idir / f"{image_file.stem}{suffix}")
                    copy_file(source_file,
                            self.target_dir / set_name / idir / source_file.name
                    )

    def __call__(self, config, iteration):
        # home_iteration = iteration # // len(self.prefixes)
        room_iteration = iteration # % len(self.prefixes)
        config = super().__call__(config, room_iteration)
        config["dataset"]["weights"] = {self.prefixes[room_iteration]:self.weight}
        home_name = self.prefixes[room_iteration].rsplit(self.prefix)[0]
        if self.weight_dir is not None and (self.weight_dir / home_name / "model.pt").exists():
            base_config = get_config(self.weight_dir / home_name / "config.toml")
            config["network"] = {**base_config["network"], **config["network"]}
            config["dataset"] = {**base_config["dataset"], **config["dataset"]}
            config["augmentation"] = {**base_config["augmentation"], **config.get("augmentation", {})}
            config["network"]["pretrained_weights"] = str(self.weight_dir / home_name / "model.pt")


        return config
