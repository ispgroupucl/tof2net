from __future__ import print_function
import numpy as np
import os
from os import path

import skimage.draw as draw
import cv2
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import imgaug as ia
from math import ceil
from tofnet.data.preprocess import *

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def to_tensor(pic):
    if pic.ndim == 2:
        pic = pic[:, :, None]

    return torch.from_numpy(pic.transpose((2, 0, 1)))

class Pipeline():
    """ Handles the data preprocessing 'pipeline'

    Arguments:
        fxs: Array of transformations that need to be applied

    .. note::
        Transformations can either be functions or objects
        that implement __call__ with the following interface:

            * args: img, img_type
            * returns: the appropriate transform of img

        If a Transformation has an internal state that should
        change, it MUST implement change_state
    """
    def __init__(self, fxs):
        self.fxs = fxs

    def append(self, fx):
        self.fxs.append(fx)

    def __call__(self, img, img_type, dico_imgs):
        """Applies the pipeline to img

        # Arguments:
            img:           the input image (N,C,H,W)
            img_type:      the type of img {image, mask}
        # Returns:
            The img modified according to params
        """
        for fx in self.fxs:
            img = fx(img, img_type, dico_imgs=dico_imgs)
        return img

    def change_state(self):
        """Changes the state of all the fxs randomly
        """
        for fx in self.fxs:
            if hasattr(fx, "change_state"):
                fx.change_state()


class TrainLoader(DataLoader):
    """ Handles the image augmentation

    Arguments:
        data (Dataset): torch Dataset that reads the correct data
        batch_size (int): #images to be concatenated
        pipeline (Pipeline): Pipeline object to be applied on an image batch in order
                            to perform augmentation
        shuffle (bool): wether to shuffle the dataset at each epoch or not
        num_workers (int): #parallel workers to load the data
    """
    def __init__(self, data:Dataset, inputs, outputs, batch_size=1, num_workers=4,
                    pipeline:Pipeline=None, shuffle=True, sampler=None, visualize=False):
        super().__init__(data, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, sampler=sampler,
                            **({"collate_fn": data.collate_fn} if hasattr(data, "collate_fn") else {})
                        )
        self.inputs    = inputs
        self.outputs   = outputs
        self.pipeline  = pipeline
        if visualize is not False:
            self.visualize(dir=visualize)

    def visualize(self, dir=None, subset=10, ignore_index=255):
        subset = subset or len(self)
        import matplotlib
        if dir is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import math
        mult = 5

        fig, axs  = plt.subplots(subset, len(self.outputs)+len(self.inputs), figsize=((len(self.outputs)+len(self.inputs))*mult,subset*mult))
        for ax, data in zip(axs, self): # thank you PYTHON :D
            for i, name in enumerate(self.inputs):
                if name in {"image"}:
                    ax[i].imshow(data[0][name][0,0].cpu())
                ax[i].set_title(name)
            outs = data[1]
            for i, out_class in enumerate(self.outputs):
                name = out_class.name
                if name not in {"mask", "keypoints", "class"}:
                    continue
                img = outs[i].cpu()
                if (img.dim()==4):
                    img = img[0]
                if (img.dim()==3 and img.shape[0]!=3):
                    if name=="keypoints":
                        img = torch.sum(img, dim=0)
                    else: # mask
                        img = img[0]
                    img = img.numpy()
                    new_img = np.zeros(img.shape)
                    new_img = img
                    ax[i+len(self.inputs)].imshow(new_img)
                else:
                    ax[i+len(self.inputs)].text(0.5,0.5, int(img[0]))
                ax[i+len(self.inputs)].set_title(name)
        if dir is None:
            plt.show()
        else:
            plt.savefig(dir, bbox_inches='tight')
        plt.close(fig)

    def __iter__(self):
        for imgs in super().__iter__():
            # Applies the pipeline on the img batch
            if self.pipeline is not None:
                self.pipeline.change_state()
                result = {}
                for img_type, img in imgs.items():
                    img_simple_type = img_type.split('_')[0] # FIXME
                    result[img_type] = self.pipeline(img, img_simple_type, imgs)

            # Generates the desired outputs for the network
            imgs    = result
            inputs = {}
            for dtype in self.inputs:
                inputs[dtype]  = imgs[dtype]
            targets = []
            for out in self.outputs:
                target = imgs[out]
                targets.append(target)

            yield inputs, targets

def segment_image(image, mask, colors=None, alpha=0.75):
    n_classes = mask.shape[0]
    assert n_classes < len(DEFAULT_SEGMENT_COLORS)

    # Process image
    if image.shape[0] != 3:
        image = image.expand(3,-1,-1)
    image = image.cpu()
    # Process mask
    if image.shape[-2:] != mask.shape[-2:]:
            mask =  F.interpolate(
                mask.unsqueeze(1).float(), size=image.shape[-2:], mode="nearest"
            ).squeeze(1).long()
    mask = mask.argmax(0)
    mask = DEFAULT_SEGMENT_COLORS[mask].permute(2,0,1)

    mix = (1-alpha)*image+alpha*mask
    return mix



def save_colored_masks(base_path, save_path, npyfile, input_size,
                        n_classes = 2, multi_output=False, classnames=None,
                        generator=None, dataset=None, name="mask", regr=False):

    save_path_img = save_path
    save_path_seg = save_path
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)
    if not os.path.exists(save_path_seg):
        os.makedirs(save_path_seg)

    if multi_output:
        npyfile = npyfile[-1]

    try:
        colors = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS
    except: # API change
        colors = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS

    # Draw color info
    if classnames is not None:
        patches = []
        for i in range(len(classnames)):
            color = [x/255 for x in colors[i+1]]
            classname = classnames[i]
            patches.append(Patch(color=color, label=classname))
        plt.figure(figsize=(0.1,0.1))
        plt.axis('off')
        ncol = ceil(len(classnames)/10)
        plt.legend(handles=patches, loc='center', ncol=ncol, frameon=False)
        plt.savefig(path.join(save_path, 'legend.png'), bbox_inches='tight')
        plt.close()

    # Iterate over all the
    for i,item in enumerate(npyfile):
        orig = generator.get_filenames(i)
        if type(orig) == list:
            orig = orig[0]
        out = path.join(save_path_img, f"{name}_{path.split(orig)[-1]}")
        seg = path.join(save_path_seg, f"out_{name}_{path.split(orig)[-1]}")
        img = generator[i][generator.dataset.input_type].squeeze(0).expand(3,-1,-1).numpy().transpose((1,2,0))

        # Generate class-map
        if regr is False:
            # Interpolation for better upsampling
            item = cv2.resize(
                item, (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            label_item = np.argmax(item, axis=-1)
            label = ia.SegmentationMapOnImage(label_item, shape=img.shape, nb_classes=n_classes)

            out_image = label.draw_on_image(img, background_threshold=0.01, colors=colors, alpha=0.75)

            # Get class-labels only
            if n_classes>1:
                item = np.argmax(item, axis=-1)
            else:
                item = np.round(item)

        # Generate keypoint regression
        elif regr is True:
            # Interpolate item to real image size
            item = cv2.resize(
                item, (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            summed_item = np.sum(item, axis=-1)
            alpha = 0.75

            # Generate global heatmap
            cm = matplotlib.cm.get_cmap("magma")
            cm_gray = matplotlib.cm.get_cmap("gray")
            img_gray = img # cm_gray(img)
            mask = cm(summed_item)

            # Overlay heatmap on img
            out_image = img_gray
            th = 0.5
            merge_cond = np.repeat(np.expand_dims((summed_item > th).astype(np.float), -1), 3, axis=-1)
            out_image = (
                ((1-alpha) * out_image + alpha * mask[:,:,:-1] * 255 ) * merge_cond +
                out_image * (1 - merge_cond)
            )

            # Draw keypoints
            keypoints = np.zeros(mask.shape[:2]+(3,))
            for i in range(item.shape[-1]):
                r,c = np.unravel_index(np.argmax(item[:,:,i]), item[:,:,i].shape)
                rr,cc = draw.circle(r,c,5, shape=item[:,:,i].shape)
                keypoints[rr,cc,:] = DEFAULT_SEGMENT_COLORS[i+1]
            keypoints = cv2.resize(keypoints, (img_gray.shape[1], img_gray.shape[0])).astype(np.float)*255

            # Assemble everything
            merge_cond = np.sum(keypoints, axis=-1) > 0
            out_image[merge_cond] =  (1-alpha) * out_image[merge_cond] + alpha * keypoints[merge_cond]


            # Scale network output for visualization
            item = np.clip(summed_item*255, 0, 255).astype(np.uint8)

        out_image = cv2.cvtColor(out_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(out, out_image)
        cv2.imwrite(seg, item)


DEFAULT_SEGMENT_COLORS = torch.tensor([
        (0, 0, 0),  # black
        (230, 25, 75),  # red
        (60, 180, 75),  # green
        (255, 225, 25),  # yellow
        (0, 130, 200),  # blue
        (245, 130, 48),  # orange
        (145, 30, 180),  # purple
        (70, 240, 240),  # cyan
        (240, 50, 230),  # magenta
        (210, 245, 60),  # lime
        (250, 190, 190),  # pink
        (0, 128, 128),  # teal
        (230, 190, 255),  # lavender
        (170, 110, 40),  # brown
        (255, 250, 200),  # beige
        (128, 0, 0),  # maroon
        (170, 255, 195),  # mint
        (128, 128, 0),  # olive
        (255, 215, 180),  # coral
        (0, 0, 128),  # navy
        (128, 128, 128),  # grey
        (255, 255, 255),  # white
        # --
        (115, 12, 37),  # dark red
        (30, 90, 37),  # dark green
        (127, 112, 12),  # dark yellow
        (0, 65, 100),  # dark blue
        (122, 65, 24),  # dark orange
        (72, 15, 90),  # dark purple
        (35, 120, 120),  # dark cyan
        (120, 25, 115),  # dark magenta
        (105, 122, 30),  # dark lime
        (125, 95, 95),  # dark pink
        (0, 64, 64),  # dark teal
        (115, 95, 127),  # dark lavender
        (85, 55, 20),  # dark brown
        (127, 125, 100),  # dark beige
        (64, 0, 0),  # dark maroon
        (85, 127, 97),  # dark mint
        (64, 64, 0),  # dark olive
        (127, 107, 90),  # dark coral
        (0, 0, 64),  # dark navy
        (64, 64, 64),  # dark grey
    ], dtype=torch.float) / 255