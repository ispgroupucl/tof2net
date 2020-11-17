
import random
import torch.nn.functional as F
import torch
import numpy as np


class RandomAugmentation():
    """ Abstract class for any form of augmentation. Posesses a seed-rewind mechanism
        in order for different inputs to be augmented in the exact same way.
    """
    def __init__(self):
        self.rand = random.Random()
        self.change_state()

    def change_state(self):
        self.seed = self.rand.getstate()

    def __call__(self):
        self.rand.setstate(self.seed)


class RandomScalingCropping(RandomAugmentation):
    def __init__(self, mult_range, shape):
        """ Randomly scales and crops the result to a certain shape
            (Also does some translation outside of the borders with zero-padding)

        Arguments:
            mult_range (tuple): contains (min_scaling_factor, max_scaling_factor), the minimum
                            and maximum scaling ratios
            shape (tuple): containing (x_dim, y_dim), the post-cropping dimensions
        """
        super().__init__()
        self.min, self.max = mult_range
        self.shape = shape

    def _crop(self, img):
        # Compute the indices we want to keep
        img_dim_x, img_dim_y = img.size()[-2], img.size()[-1]
        offset_x, offset_y = self.shape[0]//3, self.shape[1]//3
        center_x, center_y = self.rand.randrange(min(offset_x, img_dim_x//2), max(img_dim_x//2+1, img_dim_x-offset_x)), \
                                self.rand.randrange(min(offset_y, img_dim_y//2), max(img_dim_y//2+1, img_dim_y-offset_y))
        start_x, start_y = center_x-self.shape[0]//2, center_y-self.shape[1]//2
        end_x, end_y = start_x+self.shape[0], start_y+self.shape[1]

        # Compute necessary padding
        spad_x, spad_y = -start_x if start_x<0 else 0, -start_y if start_y<0 else 0
        ee_x, ee_y = end_x-img_dim_x, end_y-img_dim_y
        epad_x, epad_y = ee_x if ee_x>0 else 0, ee_y if ee_y>0 else 0
        img = F.pad(img, (spad_y, epad_y, spad_x, epad_x))
        start_x, end_x = start_x+spad_x, end_x+spad_x
        start_y, end_y = start_y+spad_y, end_y+spad_y

        # Extract cropped part
        img = img[:,:,start_x:end_x, start_y:end_y]
        return img

    def __call__(self, img: torch.Tensor, img_type, **kwargs):
        super().__call__()
        if img_type in {"class"}:
            return img
        res = ()
        for i in range(img.size()[0]):
            ii = img[i]
            scale_factor = self.rand.uniform(self.min, self.max)
            if img_type in {"image", "depth", "keypoints"}:
                ii = ii.unsqueeze_(0)
                ii = F.interpolate(ii, scale_factor=scale_factor, mode="bilinear", align_corners=True)
                ii = self._crop(ii)
                res += ii,
            elif img_type == "mask":
                ii = ii.unsqueeze_(0).unsqueeze_(0)
                ii = F.interpolate(ii, scale_factor=scale_factor, mode="nearest")
                ii = self._crop(ii)
                ii = ii.squeeze_(0)
                res += ii,
            else:
                raise ValueError(f"unknown type: {img_type}")
        if res == ():
            res = img
        else:
            res = torch.cat(res)
        return res

class RandomFlipping(RandomAugmentation):
    """ Augmentation module that randomly flips images

    Arguments:
        probability (float): flip probability
        direction (str): Either horizontal or vertical flipping
    """
    def __init__(self, probability, direction="horizontal"):
        super().__init__()
        self.probability = probability
        assert (direction in ["horizontal", "vertical"]), "Only supported flippings are vertical and horizontal"
        directions = {"horizontal": -1, "vertical": -2}
        self.axis = directions[direction]


    def __call__(self, img, img_type, dico_imgs, **kwargs):
        super().__call__()
        if img_type in {"class", "pcd"}:
            return img
        if img_type not in {"image", "depth", "mask", "keypoints"}:
            raise ValueError(f"unknown type: {img_type}")

        res = ()
        has_class = "class" in dico_imgs
        for i in range(img.size()[0]):
            ii = img[i]
            if self.rand.random() <= self.probability:
                cl = dico_imgs["class"][i] if has_class else 0
                ax = self.axis
                if cl==1 or cl==3:
                    ax = -1 if ax==-2 else -2
                ii = torch.flip(ii, [ax])
            res += ii.unsqueeze_(0),
        return torch.cat(res)

class RandomRotation(RandomAugmentation):
    def __init__(self, proportion):
        super().__init__()
        self.proportion = proportion

    def __call__(self, img, img_type, **kwargs):
        super().__call__()
        if img_type in ["class"]:
            return img
        if img_type not in ["image", "depth", "mask", "keypoints"]:
            raise ValueError(f"unknown type: {img_type}")
        res = ()
        for i in range(img.size()[0]):
            ii = img[i].unsqueeze(0)
            if img_type == "mask":
                ii = ii.unsqueeze_(0)
            angle = torch.tensor([self.proportion*self.rand.uniform(-np.pi, np.pi)])
            ## Option 2
            x_mid = (ii.size(2) + 1) / 2.
            y_mid = (ii.size(3) + 1) / 2.
            # Calculate rotation with inverse rotation matrix
            rot_matrix = torch.tensor([[torch.cos(angle), torch.sin(angle)],
                                        [-1.0*torch.sin(angle), torch.cos(angle)]])

            # Use meshgrid for pixel coords
            xv, yv = torch.meshgrid(torch.arange(ii.size(2)), torch.arange(ii.size(3)))
            xv = xv.contiguous()
            yv = yv.contiguous()
            src_ind = torch.cat((
                (xv.float() - x_mid).view(-1, 1),
                (yv.float() - y_mid).view(-1, 1)),
                dim=1
            )

            # Calculate indices using rotation matrix
            src_ind = torch.matmul(src_ind, rot_matrix.t())
            src_ind = torch.round(src_ind)

            src_ind += torch.tensor([[x_mid, y_mid]])

            # Set out of bounds indices to limits
            src_ind[src_ind < 0] = 0.
            src_ind[:, 0][src_ind[:, 0] >= ii.size(2)] = float(ii.size(2)) - 1
            src_ind[:, 1][src_ind[:, 1] >= ii.size(3)] = float(ii.size(3)) - 1

            im_rot = torch.zeros_like(ii)
            src_ind = src_ind.long()
            im_rot[:, :, xv.view(-1), yv.view(-1)] = ii[:, :, src_ind[:, 0], src_ind[:, 1]]
            im_rot = im_rot.view(ii.size(0), ii.size(1), ii.size(2), ii.size(3))

            if img_type == "mask":
                im_rot = im_rot.squeeze_(0)
            res += im_rot,
        return torch.cat(res)


class RandomSize(RandomAugmentation):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def __call__(self, img: torch.Tensor, img_type: str, **kwargs):
        super().__call__()
        if img_type in {"class"}:
            return img
        size = self.sizes[self.rand.randrange(len(self.sizes))]
        if img_type == "mask":
            img.unsqueeze_(0)
            img = F.interpolate(img, size=size, mode="nearest")
            img.squeeze_(0)
        elif img_type in  {"image", "depth", "keypoints"}:
            img = F.interpolate(img, size=size, mode="bilinear")
        else:
            raise ValueError(f"unknown type: {img_type}")
        return img
