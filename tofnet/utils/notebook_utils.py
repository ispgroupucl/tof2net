import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import skimage.io as io
import ipywidgets as widgets
from collections import defaultdict, OrderedDict
from itertools import chain
from skimage import measure
from IPython.display import Markdown, display, JSON
import seaborn as sns
import toml

class Logpath:
    def __init__(self, logspath, multiroom=True):
        logs = sorted([(x.name, x) for x in logspath.iterdir()], reverse=True)
        self.dropdown = widgets.Select(index=0, options=logs, rows=10, layout={'width': '70%'})
        self.value = logs[0][1]
        self.dropdown.observe(self.change)
        self.tab = widgets.Tab()
        self.taboutputs = {}
        self.rooms = Roomdirs(self.value, multiroom)
        self.vbox = widgets.HBox([self.dropdown, self.rooms.select])
        display(self.vbox)
        display(self.tab)
        self.change({"name": "value", "new":self.value})

    @property
    def config(self):
        with open(self.value / "base_config.toml") as config:
            return toml.load(config, OrderedDict)

    def change(self, modif):
        if modif["name"]=="value":
            self.value = modif['new']
            self.rooms.change_roomdirs(self.value)
            cfg = self.config
            tab = self.tab
            for c in cfg.keys():
                if hasattr(cfg[c], "items") or type(cfg[c]) is list:
                    if c not in self.taboutputs:
                        self.taboutputs[c] = widgets.Output()
                        tab.children = (*tab.children, self.taboutputs[c])
                        tab.set_title(len(list(tab.children))-1, c)
            for c in self.taboutputs:
                self.taboutputs[c].clear_output(wait=(c in cfg))
                if c not in cfg:
                    continue
                with self.taboutputs[c]:
                    if hasattr(cfg[c], "items"):
                        for key, val in cfg[c].items():
                            print(key, ":", val)
                    else:
                        for item in cfg[c]:
                            for key, val in item.items():
                                print(key, ":", val)

class Roomdirs:
    def __init__(self, logpath, multiroom=True):
        roomdirs = [d.name for d in sorted(logpath.iterdir()) if d.is_dir()]
        self.multiroom = multiroom
        if multiroom:
            self.select = widgets.SelectMultiple(value=tuple(roomdirs), options=roomdirs, rows=10, layout={'width': '30%'})
        else:
            self.select = widgets.Select(value=roomdirs[0], options=roomdirs, rows=10, layout={'width': '30%'})

        self.logpath = logpath
        self.output = widgets.Output()
        display(self.output)

    def change_roomdirs(self, logpath):
        dirs = [d.name for d in sorted(logpath.iterdir()) if d.is_dir()]
        self.select.options = dirs
        if self.multiroom:
            self.select.value = tuple(dirs)
        else:
            self.select.value = self.select.options[-1]

        self.logpath = logpath

    @property
    def value(self):
        if self.multiroom:
            return [self.logpath / x for x in self.select.value]
        else:
            return self.logpath / self.select.value

    @property
    def config(self):
        result = {}
        for value in self.value:
            with open(value / "config.toml") as config:
                result[value.name] = toml.load(config, OrderedDict)
        return result


class DataSelection:
    def __init__(self, path, description="Dataset:",  default=None, notebook=True):
        dirs = [
            d.parent.relative_to(path)
            for d in sorted(list(path.glob("*/image"))+list(path.glob("*/*/image"))) if d.is_dir()
        ]
        default = default or dirs[0]
        self.path = path
        self.value = self.path / default
        self.cached = {}
        if notebook:
            self.select = widgets.Dropdown(description=description ,value=default, options=dirs, rows=len(dirs), layout={'width': 'initial'})
            self.select.observe(self.change)
            display(self.select)
            self.dtype_display = display(self.dtypes, display_id=True)
            self.update_dtypes()

    def change(self, change):
        if change["name"]=="value":
            self.value = self.path / change["new"]
            self.update_dtypes()

    def update_dtypes(self):
        md = "**[" + ", ".join([dtype.name for dtype in self.dtypes]) + "]**"
        self.dtype_display.update(Markdown(md))

    def __getitem__(self, idx):
        result = {}
        for dtype in self.dtypes:
            if str(dtype) in self.cached:
                result[dtype.stem] = dtype/f"{idx}{self.cached[str(dtype)]}"
                continue
            for val in dtype.glob(f"{idx}.*"):
                self.cached[str(dtype)] = val.suffix
                result[dtype.stem] = val # next(self.value.glob(f"**/{dtype}/{idx}.*"))
        return result

    def iter(self, prefix="", exclude=None, sample=None):
        return self.__iter__(prefix, exclude, sample)

    def __iter__(self, prefix="", exclude=None, sample=None):
        for img in self.image_names(prefix, exclude, sample):
            yield self[img]

    @property
    def dtypes(self):
        dtypes =  set()
        for imagedir in chain(self.value.glob("**/image/"), self.value.glob("**/img/")):
            dtypes.update([x for x in imagedir.parent.iterdir() if x.is_dir()])
        return dtypes

    def image_names(self, prefix="", exclude=None, sample=None):
        sample_dict = defaultdict(int)
        for img in self.value.glob(f"image/{prefix}*"):
            sample_key = img.stem.rsplit("_", 1)[0]
            if sample is not None and sample_dict[sample_key] > sample:
                continue
            else:
                sample_dict[sample_key] += 1
            if exclude is None or exclude not in str(img):
                yield img.stem


class ImageSelection():
    def __init__(
        self, fpath, vis_fx=None, dtypes=("pcd", "image", "conf"), sleep=500
    ):
        # Parse args
        self.fpath = fpath
        self.subdirs = [x for x in fpath.iterdir()]
        imgs=[]
        for sdir in self.subdirs:
            imgs += [x.stem for x in self.fpath.glob(f"{str(sdir.stem)}/pcd/*.pcd")]
        self.imgs = sorted(imgs)
        self.dtypes = dtypes
        self.vis_fx = vis_fx

        # Create all widgets
        self.selected = widgets.Text(
            value=imgs[0], layout={'width': '30%'}
        )
        self.play = widgets.Play(
            value=0, min=0, max=len(imgs)-1, step=1, interval=sleep,
            disabled=False, continuous_update=False
        )
        self.slider = widgets.IntSlider(
            min=0, max=len(imgs)-1, value=0,
            continuous_update=False,
        )
        self.play.observe(self.change)
        self.slider.observe(self.change)
        widgets.jslink((self.play, "value"), (self.slider, "value"))

        vbox = widgets.VBox([self.selected, self.slider, self.play])
        display(vbox)

        # Matplotlib 'canvas' init
        if self.vis_fx is not None:
            self.main_fig = plt.figure(figsize=(6, 6))
            self.fig = self.main_fig.add_subplot(111)

        # Initial setup
        self.change({"name": "value"})


    def change(self, change):
        if change["name"] == "value":
            self.selected.value = stem = self.imgs[self.slider.value]
            for sdir in self.subdirs:
                sample = {}
                for dtype in self.dtypes:
                    fp = sdir / dtype
                    fp = list(fp.glob(f"{stem}.*"))
                    if len(fp) < 1:
                        break
                    sample[dtype] = str(fp[0])
                sample = io.read_sample(sample)
                if len(sample) == len(self.dtypes):
                    break
            else:
                print("No previz found")
                return
            if self.vis_fx is not None:
                self.fig.clear()
                sample["info"] = (sdir, stem)
                img, annot = self.vis_fx(sample)
                self.fig.imshow(img, cmap="gray")
                if annot is not None:
                    self.fig.imshow(annot, alpha=0.7)
