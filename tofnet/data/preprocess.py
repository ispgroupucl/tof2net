import cv2
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import cv2

def predict_preprocess():
    def _predict(img, img_type, **kwargs):
        if img_type in {"pcd"}:
            print(img["shape"])
            img = torch.tensor(img["points"])
            img[torch.isnan(img)] = 0
            img = img.reshape(120,160,3)
            img = torch.flip(img, dims=(0,1))
            img = img.reshape(-1, 3)
            return img.unsqueeze(0)
        else:
            imgs = to_tensor(img)
            return imgs.unsqueeze(0)
    return _predict

def to_tensor(pic):
    if pic.ndim == 2:
        pic = pic[:, :, None]

    return torch.from_numpy(pic.transpose((2, 0, 1)).copy())


def get_resize(new_size):
    def resize(imgs, img_type, **kwargs):
        if img_type in {"class", "pcd"}:
            return imgs
        elif img_type not in {"mask", "image", "depth", "keypoints", "var"}:
            raise ValueError(f"type {img_type} not known")
        if img_type == "mask":
            imgs = imgs.unsqueeze(1)
        imgs = F.interpolate(imgs, size=new_size,
                mode="bilinear" if img_type=="image" else "nearest",
                **({"align_corners":True} if img_type=="image" else {}))
        if img_type == "mask":
            imgs = imgs.squeeze(1)
        return imgs
    return resize

def get_device(device):
    def move_to_device(imgs, img_type, **kwargs):
        imgs = imgs.to(device=device, dtype=torch.float32)
        return imgs
    return move_to_device

def get_gpu():
    gpu = torch.device("cuda:0")
    def move_to_gpu(imgs, img_type, **kwargs):
        imgs = imgs.to(device=gpu, dtype=torch.float32)
        return imgs
    return move_to_gpu

def normalize(imgs, img_type, **kwargs):
    """ Normalizes differently depending on type
    """
    if img_type in {"image"}:
        return imgs/255.0
    elif img_type == "mask" or img_type == "class":
        return imgs.long()
    elif img_type in {"pcd"}:
        imgs = imgs
        return imgs
    else:
        return imgs

def crop_center(img, start_prop=0):
    y,x = img.shape[-2:]
    start = x//start_prop if start_prop!= 0 else 0
    cropx = min(x,y)
    cropy = min(x,y)
    startx = (x//2-(cropx//2)+start)
    if start < 0:
        startx = 0
    elif start > 0:
        startx = x - cropx
    starty = y//2-(cropy//2)
    if img.dim() == 4:
        return img[:,:,starty:starty+cropy,startx:startx+cropx]
    elif img.dim() == 3:
        return img[:,starty:starty+cropy,startx:startx+cropx]