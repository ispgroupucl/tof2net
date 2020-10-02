#!/bin/python3
import sys
import os
from os import path as op
from glob import glob
import cv2 as cv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import pprint

def get_frequencies(dico, _img, rcl):
    """ Computes the class frequency for an image in order to compute a per-class median-weighted
        frequency value for a whole dataset
    """
    img = cv.imread(_img, cv.IMREAD_GRAYSCALE)
    classes = np.unique(img)
    npix_tot = img.shape[-1]*img.shape[-2]

    for cl in rcl:
        npix_tot -= np.sum(img==cl)
    if npix_tot == 0:
        print(f"Error on {_img}")
        return

    for cl in [cl for cl in classes if cl not in rcl]:
        npix_cl = np.sum(img==cl)

        if npix_cl > 0:
            dico[cl]["npix"] += npix_cl
            dico[cl]["totpix"]+= npix_tot


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("./weights.py <data_directory> [<classes to neglect>]")
        exit(-1)
    dd = sys.argv[1]
    rcl = set()
    for i in range(2, len(sys.argv)):
        rcl.add(int(sys.argv[i]))
    print(f"Neglected classes {rcl}")


    name = op.join(dd, "weights.csv")

    if op.exists(op.join(dd, "labeled")):
        dd = op.join(dd, "labeled")

    if op.exists(op.join(dd, "train")):
        dd = op.join(dd, "train")

    dd = op.join(dd, "mask")
    if not op.exists(dd):
        print("Only masks are supported for now")
        exit(-1)

    dico = defaultdict(lambda: {"npix": 0, "totpix": 0})
    cnt = 0
    for img in tqdm(glob(op.join(dd+"/*"))):
        get_frequencies(dico, img, rcl=rcl)
        cnt += 1

    classes = sorted(dico.keys())
    freqs = [dico[cl]["npix"]/dico[cl]["totpix"] for cl in classes]
    med_freq = np.median(freqs)
    final = {}
    for cl, freq in zip(classes, freqs):
        final[cl] = med_freq / freq
    pprint.pprint(final)
    print([final[cl] for cl in classes])
    df = pd.DataFrame.from_dict(final, orient='index')
    df.to_csv(name)


